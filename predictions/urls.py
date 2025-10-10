from django.urls import path
from .views import ForecastPredictionsView

urlpatterns = [
    path("forecasts/<int:forecast_id>/predictions/", ForecastPredictionsView.as_view(), name="forecast_predictions"),
]