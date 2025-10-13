from rest_framework.views import APIView
from rest_framework.response import Response
from datetime import datetime
from django.utils import timezone
from .serializers import PredictionSerializer 
from .models import Forecast, Prediction

class ForecastPredictionsView(APIView):
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
                for p in model.predictions.filter(prediction_datetime__gte=now).order_by("prediction_datetime")
            ]
            result["models"].append({
                "model_name": model.name,
                "granularity": model.granularity,
                "predictions": predictions
            })

        return Response(result)
    
class PredictionListView(APIView):
    """
    Retorna uma lista de previsões, opcionalmente filtrada por uma data específica.
    """
    def get(self, request):
        # Pega o parâmetro 'date' da URL (ex: /api/predictions/?date=2025-10-13)
        date_param = request.query_params.get('date', None)
        
        queryset = Prediction.objects.all()

        if date_param:
            try:
                # Converte o texto da data para um objeto de data
                filter_date = datetime.strptime(date_param, '%Y-%m-%d').date()
                # Filtra o queryset pela data
                queryset = queryset.filter(prediction_date=filter_date)
            except ValueError:
                # Retorna um erro se o formato da data for inválido
                return Response({"error": "Formato de data inválido. Use AAAA-MM-DD."}, status=400)

        # Usa o serializer para converter os dados para JSON
        serializer = PredictionSerializer(queryset, many=True)
        return Response(serializer.data)   