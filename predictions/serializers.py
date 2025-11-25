from rest_framework import serializers
from .models import Forecast, PredictionModel, Prediction

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['prediction_date', 'prediction_time', 'predicted_occupancy']


class ForecastSerializer(serializers.ModelSerializer):
    class Meta:
        model = Forecast
        fields = '__all__'

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'

class PredictionModelSerializer(serializers.ModelSerializer):
  
    forecast_name = serializers.CharField(source='forecast.name', read_only=True)
    
    class Meta:
        model = PredictionModel
        fields = [
            'id', 
            'name', 
            'granularity', 
            'forecast', 
            'forecast_name', 
            'exog_columns'
        ]