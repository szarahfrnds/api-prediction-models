from django.contrib import admin
from .models import Forecast, PredictionModel, Prediction

@admin.register(Forecast)
class ForecastAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "description")
    search_fields = ("name",)

@admin.register(PredictionModel)
class PredictionModelAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "forecast", "granularity", "path", "created_at")
    list_filter = ("forecast", "granularity")
    search_fields = ("name", "forecast__name")

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "model", "prediction_datetime", "value", "created_at")
    list_filter = ("model__forecast", "model")
    date_hierarchy = "prediction_datetime"
    search_fields = ("model__name", "model__forecast__name")
