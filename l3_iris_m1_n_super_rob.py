'''
    ||>Esse módulo realizará as seguintes tarefas:

        Carregamento do Dataset Iris: Importa o dataset e prepara a estrutura para análise.
        Exibição das Estatísticas Descritivas: Exibe estatísticas descritivas como média, mediana, desvio padrão, valores mínimos e máximos de cada atributo.
        Verificação e Tratamento de Dados Ausentes e Outliers: Apenas para fins de demonstração, identificaremos potenciais outliers e exibição de valores ausentes.
        Padronização dos Dados: Aplicar padronização para garantir que cada atributo tenha média 0 e desvio padrão 1, tornando-o adequado para algoritmos de clustering.

    ||> Importante:
        Outliers na Coluna sepal width (cm):

        A detecção de outliers foi feita utilizando o IQR (Intervalo Interquartil). O IQR é calculado como a diferença entre o terceiro quartil (Q3) e o primeiro quartil (Q1).
        Qualquer valor abaixo de: Q1 - 1.5×IQR ou acima de: Q3+1.5×IQR é considerado um outlier.
        No caso de sepal width (cm), encontramos 4 outliers que caem fora desse intervalo. Esse método é sensível a distribuições que não são normais, e é comum que datasets como o Iris apresentem valores mais extremos em uma das dimensões, o que pode gerar alguns outliers.
        Padronização e Valores Negativos:

        A padronização foi feita usando o StandardScaler do scikit-learn, que transforma os dados para que cada coluna tenha média zero e desvio padrão 1.
        Isso implica que os valores padronizados estarão centrados em torno de zero, com muitos valores negativos e positivos, dependendo da distribuição original dos dados.
        Esse método facilita a aplicação de algoritmos de clustering, pois garante que todas as características tenham a mesma escala, o que é importante para métodos baseados em distância.
'''

# Primeiro módulo - Carregamento e Pré-Processamento dos Dados
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np


def carregar_dados():
    # Carregar o dataset Iris
    iris = load_iris()
    dados = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # Adicionar informações do número de linhas e colunas
    n_linhas, n_colunas = dados.shape
    return dados, n_linhas, n_colunas


def exibir_estatisticas_descritivas(dados):
    # Exibir estatísticas descritivas
    estatisticas = dados.describe()
    return estatisticas


def verificar_dados_faltantes(dados):
    # Verificar dados faltantes
    dados_faltantes = dados.isnull().sum()
    return dados_faltantes

# Qualquer valor abaixo de: Q1 - 1.5×IQR ou acima de: Q3+1.5×IQR é considerado um outlier.


def detectar_outliers(dados):
    # Detecção de outliers usando o IQR, com indicação das linhas específicas dos outliers
    Q1 = dados.quantile(0.25)
    Q3 = dados.quantile(0.75)
    IQR = Q3 - Q1
    outliers = {}

    for coluna in dados.columns:
        # Identificar linhas onde ocorrem outliers para cada coluna
        is_outlier = (dados[coluna] < (Q1[coluna] - 1.5 * IQR[coluna])
                      ) | (dados[coluna] > (Q3[coluna] + 1.5 * IQR[coluna]))
        # Guardar apenas os valores onde há outliers
        outliers[coluna] = dados[coluna][is_outlier]

    return outliers


def padronizar_dados(dados):
    # Aplicar padronização nos atributos
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(dados)
    dados_padronizados = pd.DataFrame(
        dados_padronizados, columns=dados.columns)
    return dados_padronizados


# Integração das funções e exibição dos resultados
if __name__ == "__main__":
    # Carregar os dados e obter número de linhas e colunas
    dados, n_linhas, n_colunas = carregar_dados()

    print(f"Dimensões do Dataset - Linhas: {n_linhas}, Colunas: {n_colunas}")

    # Exibir estatísticas descritivas
    estatisticas = exibir_estatisticas_descritivas(dados)
    print("\nEstatísticas Descritivas:\n", estatisticas)

    # Verificar dados faltantes
    dados_faltantes = verificar_dados_faltantes(dados)
    print("\nDados Faltantes:\n", dados_faltantes)

    # Detectar outliers e exibir suas localizações específicas
    outliers = detectar_outliers(dados)
    print("\nOutliers Detectados (com localização):")
    for coluna, valores_outliers in outliers.items():
        if not valores_outliers.empty:
            print(f"\n{coluna}:\n", valores_outliers)
        else:
            print(f"\n{coluna}: Nenhum outlier detectado.")

    # Padronizar dados
    dados_padronizados = padronizar_dados(dados)
    print("\nDados Padronizados (primeiras linhas):\n", dados_padronizados.head())
