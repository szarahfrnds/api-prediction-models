from rest_framework import serializers
from .models import Prediction

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['prediction_date', 'prediction_time', 'predicted_occupancy']