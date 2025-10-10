import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist


def prepare_future_exog(
    future_dates: pd.DatetimeIndex,
    exog_columns: List[str],
    exog_rules: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    VERSÃO CORRIGIDA: Trata corretamente a verificação de feriados.
    """
    if not exog_columns:
        return pd.DataFrame()

    exog_df = pd.DataFrame(index=future_dates)

    exog_df['mes'] = exog_df.index.month
    exog_df['dia_do_mes'] = exog_df.index.day
    exog_df['semana_do_ano'] = exog_df.index.isocalendar().week.astype(int)
    exog_df['eh_fim_de_semana'] = (exog_df.index.dayofweek >= 5).astype(int)
    exog_df['eh_horario_almoco'] = ((exog_df.index.hour >= 12) & (exog_df.index.hour <= 14)).astype(int)

    # --- 2. Criação de features cíclicas (Sin/Cos) ---
    exog_df['hora_sin'] = np.sin(2 * np.pi * exog_df.index.hour / 24)
    exog_df['hora_cos'] = np.cos(2 * np.pi * exog_df.index.hour / 24)
    exog_df['dia_semana_sin'] = np.sin(2 * np.pi * exog_df.index.dayofweek / 7)
    exog_df['dia_semana_cos'] = np.cos(2 * np.pi * exog_df.index.dayofweek / 7)

    # --- 3. Aplicação de regras manuais (feriados) ---
    if exog_rules:
        for col, rule in exog_rules.items():
            if col not in exog_df.columns:
                exog_df[col] = 0
            if isinstance(rule, list):
                rule_dates = pd.to_datetime(rule).date
                
                # AQUI ESTÁ A CORREÇÃO: Envolvemos o array em um pd.Series
                is_holiday = pd.Series(exog_df.index.date).isin(rule_dates)
                exog_df.loc[is_holiday.values, col] = 1

    # Garante que todas as colunas esperadas existam
    for col in exog_columns:
        if col not in exog_df.columns:
            exog_df[col] = 0

    return exog_df[exog_columns]


def generate_and_save_predictions(model_id: int, prediction_days: int = 64):
    from .models import PredictionModel, Prediction

    try:
        model_obj = PredictionModel.objects.get(id=model_id)
    except ObjectDoesNotExist:
        print(f"Erro: Modelo com ID {model_id} não encontrado.")
        return

    try:
        model = joblib.load(model_obj.path)
    except FileNotFoundError:
        print(f"Erro: Arquivo do modelo não encontrado no caminho: {model_obj.path}")
        return
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    start_date = datetime.now() + timedelta(days=1)
    future_dates = pd.date_range(
        start=start_date, periods=prediction_days, freq=model_obj.granularity
    )

    exog_df = None
    if model_obj.exog_columns:
        exog_df = prepare_future_exog(
            future_dates, model_obj.exog_columns, model_obj.exog_rules
        )

    try:
        if hasattr(model, 'get_forecast'):
            forecast = model.get_forecast(steps=prediction_days, exog=exog_df)
            predictions = forecast.predicted_mean
        elif hasattr(model, 'predict'):
            predictions = model.predict(exog_df)
        else:
            print(f"Erro: O modelo de '{model_obj.path}' não tem método 'get_forecast' ou 'predict'.")
            return
    except Exception as e:
        print(f"Erro durante a geração da previsão: {e}")
        return

    with transaction.atomic():
        predictions_to_create = []
        for dt, value in zip(future_dates, predictions):
            predictions_to_create.append(
                Prediction(
                    model=model_obj,
                    prediction_datetime=dt,
                    value=float(np.round(value, 2))
                )
            )
        Prediction.objects.bulk_create(predictions_to_create, ignore_conflicts=True)

    print(f"Previsões para o modelo '{model_obj.name}' geradas e salvas com sucesso.")