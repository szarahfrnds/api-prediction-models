from django.urls import path
from .views import ForecastPredictionsView, PredictionListView, OnDemandPredictionView
from .views import ForecastPredictionsView

urlpatterns = [
    path("forecasts/<int:forecast_id>/predictions/", ForecastPredictionsView.as_view(), name="forecast_predictions"),
    path("predictions/", PredictionListView.as_view(), name="prediction_list"),
    path("forecasts/<int:forecast_id>/predict/", OnDemandPredictionView.as_view(), name="on_demand_prediction"),
]