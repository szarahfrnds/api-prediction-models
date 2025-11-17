from django.db import models

class Forecast(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, null=True)
    entity_id = models.CharField(max_length=255, blank=True, null=True, help_text="O ID da entidade no banco de telemetria externo (ex: DB_DIOS_SQL)")

    def __str__(self):
        return self.name

class PredictionModel(models.Model):
    MODEL_TYPE_CHOICES = [
        ('prophet', 'Prophet'),
        ('lgbm', 'LightGBM'),
        ('sarimax', 'SARIMAX'),
        ('outro', 'Outro'), 
    ]

    forecast = models.ForeignKey(
        Forecast, on_delete=models.CASCADE, related_name="models"
    )

    model_type = models.CharField(
        max_length=50, 
        choices=MODEL_TYPE_CHOICES, 
        default='prophet'
    )
    
    name = models.CharField(max_length=100)
    path = models.CharField(max_length=255)
    granularity = models.CharField(max_length=10, default="D")
    exog_columns = models.JSONField(blank=True, null=True)
    exog_rules = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("forecast", "name")

    def __str__(self):
        return f"{self.forecast.name} - {self.name} ({self.granularity})"


class Prediction(models.Model):
    model = models.ForeignKey(
        PredictionModel,
        on_delete=models.CASCADE,
        related_name="predictions",
    )
    prediction_datetime = models.DateTimeField()
    value = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("model", "prediction_datetime")

    def __str__(self):
        return f"{self.model} - {self.prediction_datetime}: {self.value}"