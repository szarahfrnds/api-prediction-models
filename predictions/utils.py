# Em predictions/utils.py
import os
import joblib
import pandas as pd
from prophet.serialize import model_from_json
from django.conf import settings
from django.core.cache import cache

BASE_DIR = settings.BASE_DIR

def load_model_from_path(model_path, model_type):
    """
    Carrega um modelo (.pkl ou .json) do disco.
    model_path: O caminho que você digitou no Admin (ex: 'predictions/model/meu_modelo.pkl')
    model_type: O tipo que você SELECIONOU no Admin (ex: 'prophet', 'lgbm')
    """
    full_path = os.path.join(BASE_DIR, model_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Arquivo de modelo não encontrado em: {full_path}")

    # DECISÃO LIMPA: O tipo vem do banco, não do nome do arquivo
    if model_type == 'prophet' and full_path.endswith('.json'):
        print(f"Carregando modelo Prophet de: {full_path}")
        with open(full_path, 'r') as f:
            model = model_from_json(f.read())
        return model
        
    elif model_type in ['lgbm', 'sarimax'] and full_path.endswith('.pkl'):
        print(f"Carregando modelo PKL (Joblib) de: {full_path}")
        model = joblib.load(full_path)
        return model
        
    else:
        raise TypeError(f"Tipo de modelo '{model_type}' (do Admin) não é compatível com a extensão do arquivo '{full_path}'.")

def get_model_by_id(model_id):
    """
    Busca o modelo no cache ou o carrega do banco de dados (via Admin).
    """
    from .models import PredictionModel  
    
    cache_key = f"prediction_model_{model_id}"
    
    cached_model = cache.get(cache_key)
    if cached_model:
        print(f"Carregando modelo ID {model_id} do cache.")
        return cached_model

    print(f"Modelo ID {model_id} não no cache. Carregando do banco...")
    try:
        model_db = PredictionModel.objects.get(id=model_id)
        
        # --- A LÓGICA CORRETA ---
        model_path = model_db.path       # O caminho do arquivo (CharField)
        model_type = model_db.model_type # O tipo do modelo (Choices)
        # --- FIM DA LÓGICA ---

        modelo_carregado = load_model_from_path(model_path, model_type)

        if modelo_carregado:
            print(f"Modelo ID {model_id} ('{model_type}') carregado e salvo no cache.")
            cache.set(cache_key, modelo_carregado, timeout=None) 
            return modelo_carregado
            
    except PredictionModel.DoesNotExist:
        print(f"ERRO: Nenhum PredictionModel com ID {model_id} foi cadastrado no Admin.")
        return None
    except Exception as e:
        print(f"ERRO ao carregar o modelo ID {model_id}: {e}")
        return None