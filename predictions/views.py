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
from .utils import get_model_by_id, criar_features_xgboost 
import numpy as np 

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

def run_xgboost_prediction(model_db, modelo_executavel, data_inicio, data_fim):
    print(f"Rodando previsão XGBoost de {data_inicio} a {data_fim}...")
    
    future_dates = pd.date_range(start=data_inicio, end=data_fim, freq='D')
    df_future = pd.DataFrame({'ds': future_dates})
    print(f"Datas Geradas: {len(future_dates)} dias") # DEBUG
    print(f"Exemplo: {future_dates}") # DEBUG   
    
    df_processed = criar_features_xgboost(df_future)
    
    features_ordenadas = [
        'is_segunda', 'is_terca', 'is_quarta', 'is_quinta', 'is_sexta', 'is_sabado', 'is_domingo',
        'mes_sin', 'mes_cos', 'dia_mes', 'ano', 'eh_feriado'
    ]
    
    preds = modelo_executavel.predict(df_processed[features_ordenadas])
    
    df_final = df_future[['ds']].copy()
    df_final.columns = ['prediction_datetime']
    df_final['value'] = np.maximum(preds, 0).astype(int) 
    
    return df_final

def run_prophet_prediction(model_db, modelo_executavel, data_inicio, data_fim):
    granularidade = model_db.granularity 
    if granularidade not in ['H', 'D']: 
         raise ValueError(f"Prophet com granularidade '{granularidade}' não suportada. Use 'H' ou 'D'.")
         
    future_df = pd.date_range(start=data_inicio, end=data_fim, freq=granularidade)
    future_df = pd.DataFrame({'ds': future_df})
    
    if granularidade == 'H':
        future_df['weekday'] = future_df['ds'].dt.dayofweek < 5
        future_df['weekend'] = future_df['ds'].dt.dayofweek >= 5
    
    forecast = modelo_executavel.predict(future_df) 
    
    df_final = forecast[['ds', 'yhat']]
    df_final.columns = ['prediction_datetime', 'value'] 
    df_final['value'] = df_final['value'].clip(lower=0).round(2)
    return df_final


class ForecastListView(ListAPIView):
    queryset = Forecast.objects.all()
    serializer_class = ForecastSerializer

class ModelListView(ListAPIView):
    queryset = PredictionModel.objects.all()
    serializer_class = PredictionModelSerializer

class GeneratePredictionView(APIView):
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
        model_id = request.data.get('model_id') 
        data_inicio_str = request.data.get('data_inicio') 
        data_fim_str = request.data.get('data_fim')     

        if not all([model_id, data_inicio_str, data_fim_str]):
            return Response({"erro": "model_id, data_inicio, data_fim são obrigatórios."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            data_inicio_utc = pd.to_datetime(data_inicio_str)
            data_fim_utc = pd.to_datetime(data_fim_str)
            

            fuso_local = 'America/Sao_Paulo' 
            data_inicio_local_aware = data_inicio_utc.tz_convert(fuso_local)
            data_fim_local_aware = data_fim_utc.tz_convert(fuso_local)

          
            data_inicio_naive = data_inicio_local_aware.replace(tzinfo=None)
            data_fim_naive = data_fim_local_aware.replace(tzinfo=None)

            print(f"Datas recebidas (UTC): {data_inicio_str} a {data_fim_str}")
            print(f"Datas convertidas (Naive Local): {data_inicio_naive} a {data_fim_naive}")

        except Exception as e:
            return Response({"erro": f"Formato de data inválido. Use ISO 8601 (...T00:00:00Z). Erro: {e}"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            model_db = PredictionModel.objects.get(id=model_id)
        except PredictionModel.DoesNotExist:
            return Response({"erro": f"PredictionModel com ID {model_id} não encontrado."}, status=status.HTTP_404_NOT_FOUND)

        modelo_executavel = get_model_by_id(model_id)
        if modelo_executavel is None:
            return Response({"erro": f"Modelo ID {model_id} falhou ao carregar. Verifique os logs."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        model_type = model_db.model_type
        df_previsao = None
        
        try:
            if model_type == 'prophet':
                df_previsao = run_prophet_prediction(model_db, modelo_executavel, data_inicio_naive, data_fim_naive)
            elif model_type == 'xgboost':
                df_previsao = run_xgboost_prediction(model_db, modelo_executavel, data_inicio_naive, data_fim_naive)
            else:
                return Response({"erro": f"Tipo de modelo '{model_type}' não suportado pelo código."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(f"ERRO AO RODAR PREVISÃO: {e}")
            return Response({"erro": f"Erro ao rodar previsão: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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