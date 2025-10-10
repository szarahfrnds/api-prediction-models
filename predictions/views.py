from rest_framework.views import APIView
from rest_framework.response import Response
from datetime import datetime
from django.utils import timezone
from .models import Forecast

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