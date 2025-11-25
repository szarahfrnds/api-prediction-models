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


def parse_and_validate_dates(start_str, end_str):
    """Converte strings ISO para datetime naive (sem fuso) no horário local."""
    try:
        fuso_local = 'America/Sao_Paulo'
        
        d_start = pd.to_datetime(start_str).tz_convert(fuso_local).replace(tzinfo=None)
        d_end = pd.to_datetime(end_str).tz_convert(fuso_local).replace(tzinfo=None)
        
        if d_start > d_end:
            raise ValueError(f"Data Início ({d_start}) não pode ser maior que Data Fim ({d_end}).")
            
        return d_start, d_end
    except Exception as e:
        raise ValueError(f"Erro ao processar datas: {e}")

def process_prediction_task(model_db, data_inicio_naive, data_fim_naive):
    """
    Gerencia a execução: Carrega modelo -> Gera Dados -> Limpa Duplicatas -> Salva.
    Retorna a quantidade de registros criados.
    """
    modelo_executavel = get_model_by_id(model_db.id)
    if modelo_executavel is None:
        raise Exception(f"Não foi possível carregar o modelo ID {model_db.id}")

    model_type = model_db.model_type
    df_previsao = None

    if model_type == 'prophet':
        df_previsao = run_prophet_prediction(model_db, modelo_executavel, data_inicio_naive, data_fim_naive)
    elif model_type == 'xgboost':
        df_previsao = run_xgboost_prediction(model_db, modelo_executavel, data_inicio_naive, data_fim_naive)
    else:
        raise ValueError(f"Tipo '{model_type}' não suportado.")

    count_salvo = 0
    if df_previsao is not None and not df_previsao.empty:
        previsoes_para_salvar = []
        for index, row in df_previsao.iterrows():
            previsoes_para_salvar.append(
                Prediction(
                    model=model_db,
                    prediction_datetime=row['prediction_datetime'],
                    value=row['value']
                )
            )
        
        with transaction.atomic():
            Prediction.objects.filter(
                model=model_db, 
                prediction_datetime__range=(data_inicio_naive, data_fim_naive)
            ).delete()
            
            Prediction.objects.bulk_create(previsoes_para_salvar)
            count_salvo = len(previsoes_para_salvar)
            
    return count_salvo

def run_xgboost_prediction(model_db, modelo_executavel, data_inicio, data_fim):
    print(f"Rodando previsão XGBoost de {data_inicio} a {data_fim}...")
    
    future_dates = pd.date_range(start=data_inicio, end=data_fim, freq='D')
    df_future = pd.DataFrame({'ds': future_dates})
    print(f"Datas Geradas: {len(future_dates)} dias")
    print(f"Exemplo: {future_dates}")  
    
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
    """
    POST: Força a geração de previsões (útil para regerar dados manualmente ou corrigir erros)
    """
    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['model_id', 'data_inicio', 'data_fim'],
            properties={
                'model_id': openapi.Schema(type=openapi.TYPE_INTEGER),
                'data_inicio': openapi.Schema(type=openapi.TYPE_STRING, format=openapi.FORMAT_DATETIME),
                'data_fim': openapi.Schema(type=openapi.TYPE_STRING, format=openapi.FORMAT_DATETIME),
            },
        )
    )
    def post(self, request, *args, **kwargs):
        try:
            model_id = request.data.get('model_id')
            start_str = request.data.get('data_inicio')
            end_str = request.data.get('data_fim')
            
            if not all([model_id, start_str, end_str]):
                return Response({"erro": "Campos model_id, data_inicio e data_fim são obrigatórios."}, status=status.HTTP_400_BAD_REQUEST)

            start_naive, end_naive = parse_and_validate_dates(start_str, end_str)
            
            model_db = PredictionModel.objects.get(id=model_id)
            
            qtd = process_prediction_task(model_db, start_naive, end_naive)
            
            return Response(
                {"status": "Processamento concluído", "registros_gerados": qtd, "forecast_id": model_db.forecast.id},
                status=status.HTTP_201_CREATED
            )
            
        except PredictionModel.DoesNotExist:
            return Response({"erro": "Modelo não encontrado."}, status=status.HTTP_404_NOT_FOUND)
        except ValueError as ve:
            return Response({"erro": str(ve)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"erro": f"Erro interno: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

model_id_param = openapi.Parameter('model_id', openapi.IN_QUERY, description="[OPCIONAL] Filtra por um ID de modelo específico", type=openapi.TYPE_INTEGER)
start_date_param = openapi.Parameter('start_date', openapi.IN_QUERY, description="[OPCIONAL] Data/hora de início (ISO 8601)", type=openapi.TYPE_STRING, format=openapi.FORMAT_DATETIME)
end_date_param = openapi.Parameter('end_date', openapi.IN_QUERY, description="[OPCIONAL] Data/hora de fim (ISO 8601)", type=openapi.TYPE_STRING, format=openapi.FORMAT_DATETIME)

class ForecastResultView(ListAPIView):
    """
    GET: Busca previsões. 
    FUNCIONALIDADE INTELIGENTE: Se os dados não existirem no banco para o período, gera automaticamente.
    """
    serializer_class = PredictionSerializer

    @swagger_auto_schema(
        manual_parameters=[model_id_param, start_date_param, end_date_param]
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        forecast_id = self.kwargs.get('forecast_id')
        
        model_id = self.request.query_params.get('model_id')
        start_str = self.request.query_params.get('start_date')
        end_str = self.request.query_params.get('end_date')

        queryset = Prediction.objects.filter(model__forecast_id=forecast_id)

        if model_id:
            queryset = queryset.filter(model_id=model_id)

        if start_str and end_str:
            try:
                start_naive, end_naive = parse_and_validate_dates(start_str, end_str)
                
                queryset = queryset.filter(
                    prediction_datetime__gte=start_naive,
                    prediction_datetime__lte=end_naive
                )
              
                if model_id:
                    count_existente = queryset.count()
                    
                    dias_pedidos = (end_naive - start_naive).days + 1
                    
                    if count_existente < dias_pedidos:
                        print(f"⚠️ LAZY LOAD: Faltam dados (Tem {count_existente}, Precisa {dias_pedidos}). Gerando...")
                        
                        model_db = PredictionModel.objects.get(id=model_id)
                        process_prediction_task(model_db, start_naive, end_naive)
                        
                        queryset = Prediction.objects.filter(
                            model__forecast_id=forecast_id,
                            model_id=model_id,
                            prediction_datetime__gte=start_naive,
                            prediction_datetime__lte=end_naive
                        )
            except Exception as e:
                
                print(f"Erro no Lazy Loading: {e}")
                pass
        
        return queryset.order_by('prediction_datetime')