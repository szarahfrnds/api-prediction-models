from django.urls import path
from .views import ForecastPredictionsView, PredictionListView, OnDemandPredictionView, BatchPredictionView

urlpatterns = [
    path("forecasts/<int:forecast_id>/predictions/", ForecastPredictionsView.as_view(), name="forecast_predictions"),
    path("predictions/", PredictionListView.as_view(), name="prediction_list"),
    path("forecasts/<int:forecast_id>/predict/", OnDemandPredictionView.as_view(), name="on_demand_prediction"),
    path("forecasts/<int:forecast_id>/predict_batch/", BatchPredictionView.as_view(), name="batch_prediction"),
]