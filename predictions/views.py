# Em predictions/views.py
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.generics import ListAPIView
from rest_framework import status
from .models import PredictionModel, Prediction, Forecast
from .serializers import PredictionSerializer, ForecastSerializer, PredictionModelSerializer
from .utils import get_model_by_id 
from django.shortcuts import get_object_or_404
from django.db import transaction 
import holidays

# Imports do Swagger
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

# --- LÓGICA DE EXECUÇÃO DE CADA MODELO ---
# (ASSINATURA PADRÃO: (model_db, modelo_executavel, ...))

def run_prophet_prediction(model_db, modelo_executavel, data_inicio, data_fim):
    """ Gera previsão para o Prophet """
    granularidade = model_db.granularity # Pega 'H' ou 'D' do Admin
    if granularidade not in ['H', 'D']: 
         raise ValueError(f"Prophet com granularidade '{granularidade}' não suportada. Use 'H' ou 'D'.")
         
    future_df = pd.date_range(start=data_inicio, end=data_fim, freq=granularidade)
    future_df = pd.DataFrame({'ds': future_df})
    
    # Adiciona features SÓ se for horário
    if granularidade == 'H':
        future_df['weekday'] = future_df['ds'].dt.dayofweek < 5
        future_df['weekend'] = future_df['ds'].dt.dayofweek >= 5
    
    # O 'modelo_executavel' (o .json carregado) é usado aqui
    forecast = modelo_executavel.predict(future_df) 
    
    df_final = forecast[['ds', 'yhat']]
    df_final.columns = ['prediction_datetime', 'value'] 
    df_final['value'] = df_final['value'].clip(lower=0).round(2)
    return df_final


def run_sarimax_prediction(model_db, modelo_executavel, data_inicio, data_fim):
 
    granularidade = model_db.granularity 
    print(f"Rodando previsão SARIMAX de {data_inicio} a {data_fim}...")
    
    exog_features = model_db.exog_columns
    if not exog_features:
        raise ValueError("SARIMAX falhou: O campo 'exog_columns' não está preenchido no Admin.")

    future_df_range = pd.date_range(start=data_inicio, end=data_fim, freq=granularidade)
    future_df = pd.DataFrame(index=future_df_range)

    
    print("AVISO: Criando feature 'IsHolidayOrBridge' apenas como 'IsHoliday'.")
    br_holidays = holidays.Brazil(years=[future_df_range.year.min(), future_df_range.year.max()])
    
 
    is_holiday = pd.Series(future_df.index.date, index=future_df.index).isin(br_holidays) 

    future_df['IsHolidayOrBridge'] = is_holiday.astype(int)

    day_of_week = future_df.index.dayofweek 
    future_df['DiaSemana_Segunda'] = (day_of_week == 0).astype(int)
    future_df['DiaSemana_Terca']   = (day_of_week == 1).astype(int)
    future_df['DiaSemana_Quarta']  = (day_of_week == 2).astype(int)
    future_df['DiaSemana_Quinta']  = (day_of_week == 3).astype(int)
    future_df['DiaSemana_Sexta']   = (day_of_week == 4).astype(int)
    future_df['DiaSemana_Sabado']  = (day_of_week == 5).astype(int)
    future_df['DiaSemana_Domingo'] = (day_of_week == 6).astype(int)
    
    try:
        future_exog_df = future_df[exog_features] 
    except KeyError as e:
        raise ValueError(f"SARIMAX falhou: A feature {e} está em 'exog_columns', mas não foi criada na função 'run_sarimax_prediction'.")

    n_periods = len(future_exog_df)

    forecast_series = modelo_executavel.get_forecast(
        steps=n_periods,
        exog=future_exog_df
    ).predicted_mean
    
    df_final = forecast_series.to_frame(name='value')
    df_final.index = future_exog_df.index  
    df_final = df_final.reset_index().rename(columns={'index': 'prediction_datetime'})
    df_final['value'] = df_final['value'].clip(lower=0).round(2)
    
    print("Previsão SARIMAX gerada com sucesso.")
    return df_final


class ForecastListView(ListAPIView):
    queryset = Forecast.objects.all()
    serializer_class = ForecastSerializer

class ModelListView(ListAPIView):
    queryset = PredictionModel.objects.all()
    serializer_class = PredictionModelSerializer

# --- VIEW DE GERAÇÃO (O POST) ---
class GeneratePredictionView(APIView):
    """
    POST /api/predict/
    Gera uma nova previsão sob demanda e a salva no banco de dados.
    """
    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['model_id', 'data_inicio', 'data_fim'],
            properties={
                'model_id': openapi.Schema(type=openapi.TYPE_INTEGER, description='O ID do PredictionModel cadastrado no Admin.'),
                'data_inicio': openapi.Schema(type=openapi.TYPE_STRING, format=openapi.FORMAT_DATETIME, description='Início (ex: 2025-11-13T15:00:00Z)'),
                'data_fim': openapi.Schema(type=openapi.TYPE_STRING, format=openapi.FORMAT_DATETIME, description='Fim (ex: 2025-11-14T03:00:00Z)'),
            },
        ),
        responses={ 201: "{'status': 'Previsões geradas...', 'forecast_id': 1}"}
    )
    @transaction.atomic 
    def post(self, request, *args, **kwargs):
        # 1. Pega os parâmetros do front-end
        model_id = request.data.get('model_id') 
        data_inicio_str = request.data.get('data_inicio') # Pegar como string
        data_fim_str = request.data.get('data_fim')     # Pegar como string

        if not all([model_id, data_inicio_str, data_fim_str]):
            return Response({"erro": "model_id, data_inicio, data_fim são obrigatórios."}, status=status.HTTP_400_BAD_REQUEST)

        # --- [INÍCIO DA CORREÇÃO DE FUSO HORÁRIO] ---
        try:
            # 1. Converte a string (aware, em UTC/Z) para um objeto datetime (aware)
            data_inicio_utc = pd.to_datetime(data_inicio_str)
            data_fim_utc = pd.to_datetime(data_fim_str)
            
            # 2. Converte do fuso UTC para o fuso local (onde o modelo foi treinado)
            # (Se seu servidor ou dados de treino não são de SP, mude esta string)
            fuso_local = 'America/Sao_Paulo' 
            data_inicio_local_aware = data_inicio_utc.tz_convert(fuso_local)
            data_fim_local_aware = data_fim_utc.tz_convert(fuso_local)

            # 3. Remove a informação de fuso (torna "naive")
            # Agora a data está no horário de SP e "ingênua", 
            # exatamente como o modelo espera.
            data_inicio_naive = data_inicio_local_aware.replace(tzinfo=None)
            data_fim_naive = data_fim_local_aware.replace(tzinfo=None)

            print(f"Datas recebidas (UTC): {data_inicio_str} a {data_fim_str}")
            print(f"Datas convertidas (Naive Local): {data_inicio_naive} a {data_fim_naive}")

        except Exception as e:
            return Response({"erro": f"Formato de data inválido. Use ISO 8601 (...T00:00:00Z). Erro: {e}"}, status=status.HTTP_400_BAD_REQUEST)
        # --- [FIM DA CORREÇÃO DE FUSO HORÁRIO] ---

        # 2. Busca o registro do modelo no banco
        try:
            model_db = PredictionModel.objects.get(id=model_id)
        except PredictionModel.DoesNotExist:
            return Response({"erro": f"PredictionModel com ID {model_id} não encontrado."}, status=status.HTTP_404_NOT_FOUND)

        # 3. Carrega o modelo (do cache ou do disco)
        modelo_executavel = get_model_by_id(model_id)
        if modelo_executavel is None:
            return Response({"erro": f"Modelo ID {model_id} falhou ao carregar. Verifique os logs."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        model_type = model_db.model_type
        df_previsao = None
        
        try:
            # 4. Passa os valores NAIVE (convertidos) para as funções
            if model_type == 'prophet':
                df_previsao = run_prophet_prediction(model_db, modelo_executavel, data_inicio_naive, data_fim_naive)
            elif model_type == 'lgbm':
                df_previsao = run_lgbm_prediction(model_db, modelo_executavel, data_inicio_naive, data_fim_naive)
            elif model_type == 'sarimax':
                df_previsao = run_sarimax_prediction(model_db, modelo_executavel, data_inicio_naive, data_fim_naive)
            else:
                return Response({"erro": f"Tipo de modelo '{model_type}' não suportado pelo código."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(f"ERRO AO RODAR PREVISÃO: {e}")
            return Response({"erro": f"Erro ao rodar previsão: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 5. Salva no banco (o seu fluxo POST -> GET)
        previsoes_para_salvar = []
        for index, row in df_previsao.iterrows():
            previsoes_para_salvar.append(
                Prediction(
                    model=model_db,
                    prediction_datetime=row['prediction_datetime'],
                    value=row['value']
                )
            )
        
        Prediction.objects.filter(model=model_db).delete() 
        Prediction.objects.bulk_create(previsoes_para_salvar)
        
        print(f"Sucesso! {len(previsoes_para_salvar)} previsões salvas para o Modelo ID {model_id}.")
        
        return Response(
            {"status": "Previsões geradas com sucesso.", "forecast_id": model_db.forecast.id},
            status=status.HTTP_201_CREATED
        )

# --- VIEW DE LEITURA (O GET) ---
model_id_param = openapi.Parameter('model_id', openapi.IN_QUERY, description="[OPCIONAL] Filtra por um ID de modelo específico", type=openapi.TYPE_INTEGER)
start_date_param = openapi.Parameter('start_date', openapi.IN_QUERY, description="[OPCIONAL] Data/hora de início (ISO 8601)", type=openapi.TYPE_STRING, format=openapi.FORMAT_DATETIME)
end_date_param = openapi.Parameter('end_date', openapi.IN_QUERY, description="[OPCIONAL] Data/hora de fim (ISO 8601)", type=openapi.TYPE_STRING, format=openapi.FORMAT_DATETIME)

class ForecastResultView(ListAPIView):
    """
    GET /api/forecasts/<id>/
    Retorna as previsões salvas no banco para um Forecast ID.
    Permite filtrar por ?model_id=, ?start_date=, e ?end_date=
    """
    serializer_class = PredictionSerializer

    @swagger_auto_schema(
        manual_parameters=[model_id_param, start_date_param, end_date_param]
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        forecast_id = self.kwargs.get('forecast_id')
        models_in_forecast = PredictionModel.objects.filter(forecast_id=forecast_id)
        queryset = Prediction.objects.filter(model__in=models_in_forecast)

        model_id_filter = self.request.query_params.get('model_id')
        if model_id_filter:
            queryset = queryset.filter(model_id=model_id_filter)

        start_date_filter = self.request.query_params.get('start_date')
        if start_date_filter:
            queryset = queryset.filter(prediction_datetime__gte=start_date_filter)
        
        end_date_filter = self.request.query_params.get('end_date')
        if end_date_filter:
            queryset = queryset.filter(prediction_datetime__lte=end_date_filter)

        return queryset.order_by('prediction_datetime')