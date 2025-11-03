import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from django.db import transaction
from django.core.exceptions import ObjectDoesNotExist
from .models import PredictionModel, Prediction

def prepare_future_exog(
    future_dates: pd.DatetimeIndex,
    exog_columns: List[str],
    exog_rules: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Cria um DataFrame de features exógenas para datas futuras.
    Esta função é agnóstica ao modelo e gera todas as features que conhece.
    """
    if not exog_columns:
        return pd.DataFrame()

    exog_df = pd.DataFrame(index=future_dates)

    exog_df['mes'] = exog_df.index.month
    exog_df['dia_do_mes'] = exog_df.index.day
    exog_df['semana_do_ano'] = exog_df.index.isocalendar().week.astype(int)
    exog_df['eh_fim_de_semana'] = (exog_df.index.dayofweek >= 5).astype(int)
    exog_df['eh_horario_almoco'] = ((exog_df.index.hour >= 12) & (exog_df.index.hour <= 14)).astype(int)

    exog_df['hora_sin'] = np.sin(2 * np.pi * exog_df.index.hour / 24)
    exog_df['hora_cos'] = np.cos(2 * np.pi * exog_df.index.hour / 24)
    exog_df['dia_semana_sin'] = np.sin(2 * np.pi * exog_df.index.dayofweek / 7)
    exog_df['dia_semana_cos'] = np.cos(2 * np.pi * exog_df.index.dayofweek / 7)

    for i in range(7):
        exog_df[f'weekday_{i}'] = (exog_df.index.dayofweek == i).astype(int)

    if exog_rules:
        for col, rule in exog_rules.items():
            if col not in exog_df.columns:
                exog_df[col] = 0
            if isinstance(rule, list):
                rule_dates = pd.to_datetime(rule).date
                is_rule_date = pd.Series(exog_df.index.date).isin(rule_dates)
                exog_df.loc[is_rule_date.values, col] = 1

    for col in exog_columns:
        if col not in exog_df.columns:
            exog_df[col] = 0

    return exog_df[exog_columns]


def generate_single_prediction(model_id: int, prediction_datetime: pd.Timestamp, external_features: dict = None) -> float:

    model_obj = PredictionModel.objects.get(id=model_id)
    model = joblib.load(model_obj.path)

    future_date = pd.DatetimeIndex([prediction_datetime])

    exog_df = prepare_future_exog(
        future_date, model_obj.exog_columns, model_obj.exog_rules
    )

    if external_features:
        for feature_name, value in external_features.items():
            if feature_name in exog_df.columns:
                exog_df[feature_name] = value

    ordered_exog_df = exog_df[model_obj.exog_columns]

    if hasattr(model, 'get_forecast'):
        forecast = model.get_forecast(steps=1, exog=ordered_exog_df)
        prediction_value = forecast.predicted_mean.iloc[0]
    elif hasattr(model, 'predict'):
        prediction_value = model.predict(ordered_exog_df)[0]
    else:
        raise TypeError("O modelo não é compatível com os métodos 'predict' ou 'get_forecast'.")

    return float(np.round(prediction_value, 2))


def generate_and_save_predictions(model_id: int, prediction_days: int = 64):
    """
    Gera previsões em lote e salva no banco de dados. Usado pelo comando de gerenciamento.
    """
    from .models import PredictionModel, Prediction

    try:
        model_obj = PredictionModel.objects.get(id=model_id)
    except ObjectDoesNotExist:
        print(f"Erro: Modelo com ID {model_id} não encontrado.")
        return

    try:
        model = joblib.load(model_obj.path)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    freq = model_obj.granularity.lower()
    start_date = datetime.now() + timedelta(days=1)
    future_dates = pd.date_range(
        start=start_date, periods=prediction_days, freq=freq
    )

    # Esta abordagem preencherá features defasadas com 0, o que é uma limitação
    # da previsão em lote para modelos complexos.
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
            raise TypeError("O modelo não é compatível.")
    except Exception as e:
        print(f"Erro durante a geração da previsão em lote: {e}")
        return

    with transaction.atomic():
        Prediction.objects.filter(model=model_obj, prediction_datetime__gte=start_date).delete()
        predictions_to_create = [
            Prediction(
                model=model_obj,
                prediction_datetime=dt,
                value=float(np.round(value, 2))
            ) for dt, value in zip(future_dates, predictions)
        ]
        Prediction.objects.bulk_create(predictions_to_create, ignore_conflicts=True)

    print(f"Previsões em lote para o modelo '{model_obj.name}' geradas e salvas com sucesso.")