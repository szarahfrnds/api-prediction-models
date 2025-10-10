import joblib

# Coloque o nome exato do seu arquivo de modelo aqui
model_filename = 'teste.joblib'

try:
    # Carrega o modelo a partir do arquivo
    model_results = joblib.load(model_filename)

    # A maioria dos modelos statsmodels armazena os nomes das variáveis exógenas aqui
    # O atributo exog_names guarda o nome das colunas que você usou no treino
    exog_names = model_results.model.exog_names

    print("As variáveis exógenas (exog) usadas no modelo são:")
    print(exog_names)

except FileNotFoundError:
    print(f"Erro: O arquivo '{model_filename}' não foi encontrado.")
    print("Por favor, certifique-se de que o nome do arquivo está correto e no mesmo diretório do script.")
except AttributeError:
    print("Não foi possível encontrar o atributo 'exog_names' no modelo.")
    print("Isso pode significar que o modelo foi treinado sem variáveis exógenas (um modelo SARIMA em vez de SARIMAX).")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")