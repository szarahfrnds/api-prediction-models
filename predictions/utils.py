# Em predictions/utils.py
import os
import joblib
import pandas as pd
from prophet.serialize import model_from_json
from django.conf import settings
from django.core.cache import cache
import numpy as np # 
import xgboost as xgb

BASE_DIR = settings.BASE_DIR

def load_model_from_path(model_path, model_type):
  
    full_path = os.path.join(BASE_DIR, model_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Arquivo de modelo não encontrado em: {full_path}")

    if model_type == 'prophet' and full_path.endswith('.json'):
        print(f"Carregando modelo Prophet de: {full_path}")
        with open(full_path, 'r') as f:
            model = model_from_json(f.read())
        return model
        
    elif model_type in ['lgbm', 'sarimax', 'xgboost'] and full_path.endswith('.pkl'):
        print(f"Carregando modelo PKL (Joblib) de: {full_path}")
        model = joblib.load(full_path)
        return model
    
    else:
        raise TypeError(f"Tipo de modelo '{model_type}' (do Admin) não é compatível com a extensão do arquivo '{full_path}'.")

def get_model_by_id(model_id):
   
    from .models import PredictionModel  
    
    cache_key = f"prediction_model_{model_id}"
    
    cached_model = cache.get(cache_key)
    if cached_model:
        print(f"Carregando modelo ID {model_id} do cache.")
        return cached_model

    print(f"Modelo ID {model_id} não no cache. Carregando do banco...")
    try:
        model_db = PredictionModel.objects.get(id=model_id)
        
        model_path = model_db.path       
        model_type = model_db.model_type 

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

def criar_features_xgboost(df_input):
    """
    Recria as features exatas que o modelo XGBoost aprendeu no treinamento.
    """
    df = df_input.copy()
    
    if not np.issubdtype(df['ds'].dtype, np.datetime64):
        df['ds'] = pd.to_datetime(df['ds'])

    df['dia_semana'] = df['ds'].dt.dayofweek
    df['dia_mes'] = df['ds'].dt.day
    df['mes'] = df['ds'].dt.month
    df['ano'] = df['ds'].dt.year
    
    df['mes_sin'] = np.sin(2 * np.pi * df['mes']/12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes']/12)
    
    df['is_segunda'] = (df['dia_semana'] == 0).astype(int)
    df['is_terca']   = (df['dia_semana'] == 1).astype(int)
    df['is_quarta']  = (df['dia_semana'] == 2).astype(int)
    df['is_quinta']  = (df['dia_semana'] == 3).astype(int)
    df['is_sexta']   = (df['dia_semana'] == 4).astype(int)
    df['is_sabado']  = (df['dia_semana'] == 5).astype(int)
    df['is_domingo'] = (df['dia_semana'] == 6).astype(int)
    
    feriados = [
        '2024-01-01', '2024-02-12', '2024-02-13', '2024-03-29', '2024-04-21',
        '2024-05-01', '2024-05-30', '2024-09-07', '2024-10-12', '2024-11-02',
        '2024-11-15', '2024-11-20', '2024-12-25', 
        '2025-01-01', '2025-03-03', '2025-03-04', '2025-04-18', '2025-04-21',
        '2025-05-01', '2025-06-19', '2025-09-07', '2025-10-12', '2025-11-02',
        '2025-11-15', '2025-11-20', '2025-12-25'
    ]
    feriados_dt = pd.to_datetime(feriados).normalize()
    df['eh_feriado'] = df['ds'].dt.normalize().isin(feriados_dt).astype(int)
    
    return df