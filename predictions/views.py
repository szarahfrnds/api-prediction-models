from rest_framework.views import APIView
from rest_framework.response import Response
from datetime import datetime
from django.utils import timezone
import pandas as pd
from .models import Forecast, Prediction, PredictionModel
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .utils import generate_single_prediction
from django.db import transaction
from decimal import Decimal 
import pyodbc  
import json   


class OnDemandPredictionView(APIView):
    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter('period', openapi.IN_QUERY, description="Data/hora para a previsão (formato ISO 8601, ex: 2025-10-27T15:00:00)", type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('granularity', openapi.IN_QUERY, description="Granularidade do modelo ('D' para diário, 'h' para horário)", type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('ocupacao_hora_anterior', openapi.IN_QUERY, description="[Opcional] Valor da ocupação na hora anterior (necessário para modelos horários)", type=openapi.TYPE_NUMBER),
            openapi.Parameter('ocupacao_dia_anterior', openapi.IN_QUERY, description="[Opcional] Valor da ocupação na mesma hora do dia anterior (necessário para modelos horários)", type=openapi.TYPE_NUMBER),
        ]
    )
    def get(self, request, forecast_id):
        period_str = request.query_params.get('period')
        granularity = request.query_params.get('granularity')

        if not period_str or not granularity:
            return Response({"error": "Parâmetros 'period' e 'granularity' são obrigatórios."}, status=400)

        try:
            prediction_datetime = pd.to_datetime(period_str)
        except ValueError:
            return Response({"error": "Formato de 'period' inválido. Use ISO 8601 (ex: 2025-10-27T15:00:00)."}, status=400)

        try:
            model = PredictionModel.objects.get(forecast_id=forecast_id, granularity=granularity.lower())
        except PredictionModel.DoesNotExist:
            return Response({"error": f"Modelo com granularidade '{granularity}' não encontrado."}, status=404)

        external_features = {}
        supported_external_features = ['ocupacao_hora_anterior', 'ocupacao_dia_anterior']

        for feature in supported_external_features:
            if feature in model.exog_columns:
                value_str = request.query_params.get(feature)
                if value_str is None:
                    return Response({"error": f"Este modelo requer o parâmetro de URL '{feature}'."}, status=400)
                try:
                    external_features[feature] = float(value_str)
                except (ValueError, TypeError):
                    return Response({"error": f"O valor de '{feature}' deve ser um número."}, status=400)

        try:
            prediction_value = generate_single_prediction(model.id, prediction_datetime, external_features)
        except Exception as e:
            return Response({"error": f"Erro ao gerar a previsão: {str(e)}"}, status=500)
        
        Prediction.objects.update_or_create(
            model=model,
            prediction_datetime=prediction_datetime,
            defaults={'value': prediction_value}
        )

        return Response({
            "forecast_name": model.forecast.name,
            "model_name": model.name,
            "requested_period": prediction_datetime.isoformat(),
            "provided_features": external_features if external_features else "Nenhuma",
            "prediction": prediction_value
        })


class ForecastPredictionsView(APIView):
    """
    (Legado) Retorna previsões em lote que foram pré-calculadas e salvas no banco.
    """
    def get(self, request, forecast_id):
        try:
            forecast = Forecast.objects.prefetch_related("models__predictions").get(id=forecast_id)
        except Forecast.DoesNotExist:
            return Response({"error": "Forecast não encontrado"}, status=404)

        granularity_filter = request.query_params.get('granularity')
        models_query = forecast.models.all()
        if granularity_filter:
            models_query = models_query.filter(granularity=granularity_filter)

        result = {
            "forecast_name": forecast.name,
            "models": []
        }
        now = timezone.now()

        for model in models_query:
            predictions = [
                {
                    "datetime": p.prediction_datetime.isoformat(),
                    "value": p.value
                }
                for p in model.predictions.all().order_by("prediction_datetime")
            ]
            result["models"].append({
                "model_name": model.name,
                "granularity": model.granularity,
                "predictions": predictions
            })
        return Response(result)



class PredictionListView(APIView):
    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter(
                'date', 
                openapi.IN_QUERY, 
                description="Filtra as previsões pela data. Use o formato AAAA-MM-DD.",
                type=openapi.TYPE_STRING
            )
        ]
    )
    def get(self, request):
        date_param = request.query_params.get('date', None)
        queryset = Prediction.objects.all()

        if date_param:
            try:
                filter_date = datetime.strptime(date_param, '%Y-%m-%d').date()
               
                queryset = queryset.filter(prediction_date=filter_date) 
            except ValueError:
                return Response({"error": "Formato de data inválido. Use AAAA-MM-DD."}, status=400)

        return Response({"message": "Esta view precisa de um serializer configurado para funcionar."})
    


def get_external_telemetry_sum(entity_id: str, end_datetime_utc: pd.Timestamp) -> float:
    SERVER = "CA0VSQL09.br.bosch.com,56482"
    DATABASE = "DB_DIOS_SQL"
    USERNAME = "FCM"
    PASSWORD = "FCM.Campinas149"
    CONNECTION_STRING = f"DRIVER={{SQL Server}};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}"

    sql_query = """
    SELECT SUM(CAST(JSON_VALUE(data, '$.inDiff') AS INT))
    FROM Telemetry
    WHERE entity_id = ? 
      AND timestamp <= ?
    """
    
    conn = None
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        
        end_time_str = end_datetime_utc.strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute(sql_query, (entity_id, end_time_str))
        row = cursor.fetchone()
        
        if row and row[0] is not None:
            return float(row[0])
        else:
            return 0.0
            
    except Exception as e:
        raise Exception(f"Falha ao buscar dado histórico no DB_DIOS_SQL: {e}")
    finally:
        if conn:
            conn.close()


class BatchPredictionView(APIView):

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['start_date', 'end_date', 'granularity', 'initial_features'],
            properties={
                'start_date': openapi.Schema(type=openapi.TYPE_STRING, description='Data/hora de início (Formato ISO: 2025-11-10T09:00:00Z)', format='datetime'),
                'end_date': openapi.Schema(type=openapi.TYPE_STRING, description='Data/hora de fim (Formato ISO: 2025-11-17T09:00:00Z)', format='datetime'),
                'granularity': openapi.Schema(type=openapi.TYPE_STRING, description="Granularidade do modelo ('d' ou 'h')"),
                'initial_features': openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    description="[OBRIGATÓRIO PARA MODELOS 'h'] Valor inicial para 'ocupacao_hora_anterior'.",
                    properties={
                        'ocupacao_hora_anterior': openapi.Schema(type=openapi.TYPE_NUMBER),
                    }
                )
            },
        ),
        responses={201: "Previsões geradas e salvas com sucesso."}
    )
    def post(self, request, forecast_id):
        start_date_str = request.data.get('start_date')
        end_date_str = request.data.get('end_date')
        granularity = request.data.get('granularity')

        if not all([start_date_str, end_date_str, granularity]):
            return Response({"error": "Parâmetros 'start_date', 'end_date' e 'granularity' são obrigatórios."}, status=400)

        try:
            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)
        except ValueError:
            return Response({"error": "Formato de data inválido. Use ISO 8601 (ex: 2025-10-27T15:00:00Z)."}, status=400)

        try:
            model_obj = PredictionModel.objects.get(forecast_id=forecast_id, granularity=granularity.lower())
        except PredictionModel.DoesNotExist:
            return Response({"error": f"Modelo com granularidade '{granularity}' não encontrado."}, status=404)

        entity_id = model_obj.forecast.entity_id
        if not entity_id:
            return Response({"error": f"O Forecast '{model_obj.forecast.name}' não tem um 'entity_id' configurado no Admin."}, status=400)

        if start_date.tzinfo is None:
            start_date = start_date.tz_localize('UTC')
        if end_date.tzinfo is None:
            end_date = end_date.tz_localize('UTC')
        
        freq_map = {'d': 'D', 'h': 'h'}
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq_map.get(granularity.lower()))
        
        predictions_to_create = []

        try:
            with transaction.atomic():
                
                if granularity.lower() == 'd':
                    for current_date in date_range:
                        prediction_value = generate_single_prediction(model_obj.id, current_date, {})
                        predictions_to_create.append(
                            Prediction(
                                model=model_obj,
                                prediction_datetime=current_date,
                                value=prediction_value
                            )
                        )
                
                elif granularity.lower() == 'h':
                    initial_features = request.data.get('initial_features')
                    if not initial_features or 'ocupacao_hora_anterior' not in initial_features:
                        return Response({"error": "O campo 'initial_features' com 'ocupacao_hora_anterior' é obrigatório para previsões em lote horárias."}, status=400)

                    current_lag_1h = initial_features['ocupacao_hora_anterior']

                    for current_hour in date_range:
                        current_features_for_model = {}
                        
                        current_features_for_model['ocupacao_hora_anterior'] = current_lag_1h

          
                        if 'ocupacao_dia_anterior' in model_obj.exog_columns:
                            hour_minus_24 = current_hour - pd.Timedelta(days=1)
                            
                     
                            real_lag_24h = get_external_telemetry_sum(entity_id, hour_minus_24)
                            current_features_for_model['ocupacao_dia_anterior'] = real_lag_24h

                        prediction_value = generate_single_prediction(model_obj.id, current_hour, current_features_for_model)
                        
                        predictions_to_create.append(
                            Prediction(
                                model=model_obj,
                                prediction_datetime=current_hour,
                                value=prediction_value
                            )
                        )
                        
                        
                        current_lag_1h = prediction_value

                Prediction.objects.bulk_create(predictions_to_create, ignore_conflicts=True)

        except Exception as e:
            return Response({"error": f"Erro durante a geração em lote: {str(e)}"}, status=500)

        return Response({
            "message": f"{len(predictions_to_create)} previsões para o modelo '{model_obj.name}' foram geradas e salvas com sucesso."
        }, status=201)
    
