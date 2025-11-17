import joblib
import warnings
import sys

# Ignora avisos que podem aparecer ao carregar modelos
warnings.filterwarnings("ignore")

# O caminho para o seu modelo. 
# Mude se o seu script não estiver na raiz do projeto.
model_path = "predictions/model/sarimax_model_ocupacao.pkl"

try:
    # Carrega o arquivo .pkl
    model_wrapper = joblib.load(model_path)
    
    print(f"Arquivo do modelo carregado com sucesso: {model_path}")
    print(f"Tipo de objeto carregado: {type(model_wrapper)}")
    
    # Modelos SARIMAX da statsmodels geralmente são salvos como um "ResultsWrapper"
    # A lista de nomes das features (exog) está dentro do objeto .model
    
    exog_names = None

    # Tentativa 1: O caminho mais provável para um ResultsWrapper
    if hasattr(model_wrapper, '_results') and hasattr(model_wrapper._results, 'model'):
        if hasattr(model_wrapper._results.model, 'exog_names'):
            exog_names = model_wrapper._results.model.exog_names
        else:
            print("AVISO: model_wrapper._results.model não tem o atributo 'exog_names'.")

    # Tentativa 2: Se o .pkl for o próprio modelo (menos provável)
    elif hasattr(model_wrapper, 'exog_names'):
        exog_names = model_wrapper.exog_names
    
    # Tentativa 3: Se for um objeto de resultados direto
    elif hasattr(model_wrapper, 'model') and hasattr(model_wrapper.model, 'exog_names'):
         exog_names = model_wrapper.model.exog_names

    print("\n--- RESULTADO DA INSPEÇÃO ---")
    if exog_names:
        print("O modelo foi treinado com os seguintes nomes de features (exog):")
        print(exog_names)
        print("\nPróximos passos:")
        print("1. Copie esta lista EXATA.")
        print("2. Cole esta lista no campo 'exog_columns' do seu modelo no Django Admin.")
        print(f"3. Modifique 'predictions/utils.py' (função prepare_future_exog) para gerar colunas com ESSES nomes.")
    else:
        print("ERRO: Não foi possível encontrar a lista 'exog_names' automaticamente.")
        print("Por favor, verifique o tipo de objeto acima e investigue seus atributos.")

except FileNotFoundError:
    print(f"ERRO: O arquivo do modelo não foi encontrado em: {model_path}")
    print("Por favor, verifique se o caminho está correto.")
except Exception as e:
    print(f"Erro ao carregar ou inspecionar o modelo: {e}")