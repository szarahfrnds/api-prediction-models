from django.core.management.base import BaseCommand
from predictions.models import PredictionModel
from predictions.utils import generate_and_save_predictions

class Command(BaseCommand):

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.NOTICE("Iniciando o processo de geração de previsões..."))

        prediction_models = PredictionModel.objects.all()

        if not prediction_models.exists():
            self.stdout.write(self.style.WARNING("Nenhum modelo de previsão foi encontrado no banco de dados."))
            return

        success_count = 0
        error_count = 0

        for model in prediction_models:
            self.stdout.write(f"Processando modelo: {model}")
            try:
                generate_and_save_predictions(model.id)
                self.stdout.write(self.style.SUCCESS(f"-> Previsões geradas com sucesso para o modelo '{model.name}'."))
                success_count += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"-> Ocorreu um erro ao processar o modelo '{model.name}': {e}"))
                error_count += 1

        self.stdout.write("\n" + self.style.NOTICE("="*30))
        self.stdout.write(self.style.NOTICE("Processo finalizado!"))
        self.stdout.write(self.style.SUCCESS(f"Modelos processados com sucesso: {success_count}"))
        if error_count > 0:
            self.stdout.write(self.style.ERROR(f"Modelos com erro: {error_count}"))
        self.stdout.write(self.style.NOTICE("="*30))