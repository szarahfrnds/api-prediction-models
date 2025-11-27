from django.urls import path
from .views import (
    ForecastListView,
    ModelListView,
    GeneratePredictionView,
    ForecastResultView
)

urlpatterns = [
    path('forecasts/', ForecastListView.as_view(), name='forecast-list'),
    path('models/', ModelListView.as_view(), name='model-list'),
    path('predict/', GeneratePredictionView.as_view(), name='generate-prediction'),
    path('forecasts/<int:forecast_id>/predictions', ForecastResultView.as_view(), name='forecast-results'),
]