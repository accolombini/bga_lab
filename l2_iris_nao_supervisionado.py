'''
    Objetivo é demontra o uso de algoritmos de ML em aprendizado não supervisionado, no caso da identificação de clusters em dados de texto (flor de Iris).

    Estrutura do Projeto
Leitura e Carregamento dos Dados

Ler o dataset Iris (que não possui um atributo classificador explícito para o aprendizado não supervisionado).
Exibir as características dos dados: dimensões, tipos de atributos, e distribuição.
Mostrar a estrutura completa dos dados no Dashboard.
Estatística Descritiva

Exibir métricas estatísticas como: média, mediana, desvio padrão, valores mínimos/máximos.
Gerar visualizações de distribuições como histogramas e boxplots para cada atributo do dataset usando Plotly.
Divisão dos Dados

Separar os dados em 70% para treino e 30% para teste.
Avaliar a necessidade dessa divisão, já que, em aprendizado não supervisionado, muitas vezes não se trabalha diretamente com treino e teste da mesma forma que em aprendizado supervisionado.
Escolha de Algoritmos Não Supervisionados

Explorar dois algoritmos de aprendizado não supervisionado:
K-means (agrupamento baseado em centroides).
DBSCAN (agrupamento baseado em densidade, que pode lidar com ruído e dados de forma mais flexível).
Demonstrar a importância da escolha do algoritmo, comparando os resultados dos dois modelos no Dashboard.
Matriz de Confusão e Métricas de Avaliação

Gerar uma matriz de confusão (mesmo em aprendizado não supervisionado, ela pode ser gerada após atribuir rótulos aos clusters).
Calcular métricas de avaliação: precisão, recall, e F1-score. (Neste caso, teremos que associar os clusters formados pelos algoritmos com as classes verdadeiras do conjunto de dados Iris para calcular essas métricas.)
Criação do Dashboard (com Plotly e Dash)

Criar um Dashboard interativo que exiba:
Características do dataset (dimensão, atributos, etc.).
Estatísticas descritivas.
Visualizações dos dados (histogramas, boxplots).
Resultados dos modelos (clusters e matriz de confusão).

'''

# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Função para ler o conjunto de dados Iris e exibir suas características


def funcao_le_dados():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    return df, iris.target

# Função para calcular estatísticas descritivas (média, mediana, desvio padrão, etc.)


def estatisticas_descritivas(df):
    estatisticas = df.describe()
    return estatisticas

# Função para criar histogramas e boxplots usando Plotly


def criar_graficos_estatisticos(df):
    fig_histogramas = []
    fig_boxplots = []
    for column in df.columns:
        fig_histogramas.append(px.histogram(
            df, x=column, title=f"Histograma de {column}"))
        fig_boxplots.append(px.box(df, y=column, title=f"Boxplot de {column}"))
    return fig_histogramas, fig_boxplots

# Função para dividir os dados em treino (70%) e teste (30%)


def dividir_dados(df):
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    return X_train, X_test

# Função para aplicar o algoritmo K-means


def aplicar_kmeans(X, n_clusters=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans

# Função para aplicar o algoritmo DBSCAN


def aplicar_dbscan(X, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    return clusters, dbscan

# Função para aplicar PCA e reduzir os dados para 2 dimensões


def aplicar_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

# Função para visualizar os clusters usando Plotly


def visualizar_clusters(X_pca, clusters, algoritmo_nome):
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1], color=clusters,
        title=f"Clusters Identificados - {algoritmo_nome}",
        labels={'x': 'Componente Principal 1', 'y': 'Componente Principal 2'},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    return fig

# Função para criar o Dashboard com Dash e Plotly


def criar_dashboard(estatisticas, fig_histogramas, fig_boxplots,
                    fig_clusters_kmeans, fig_clusters_dbscan):
    app = Dash(__name__)

    # Extraindo a linha da média das estatísticas descritivas para o gráfico de barras
    media_df = estatisticas.loc[['mean']].transpose().reset_index()

    app.layout = html.Div([
        html.H1("Dashboard de Agrupamento Não Supervisionado - Flor de Iris"),

        # Exibir Estatísticas Descritivas
        html.H2("Média dos Atributos"),
        dcc.Graph(figure=px.bar(media_df, x='index',
                  y='mean', title="Média dos Atributos")),

        # Histogramas e Boxplots
        html.H2("Distribuições dos Dados"),
        dcc.Tabs(id="tabs-graficos", value='histogramas', children=[
            dcc.Tab(label='Histogramas', value='histogramas'),
            dcc.Tab(label='Boxplots', value='boxplots'),
        ]),
        html.Div(id='graficos-output'),

        # Visualização dos clusters - K-means
        html.H2("Clusters Identificados - K-means"),
        dcc.Graph(figure=fig_clusters_kmeans),

        # Visualização dos clusters - DBSCAN
        html.H2("Clusters Identificados - DBSCAN"),
        dcc.Graph(figure=fig_clusters_dbscan),
    ])

    @app.callback(
        Output('graficos-output', 'children'),
        [Input('tabs-graficos', 'value')]
    )
    def atualizar_graficos(tab_selecionada):
        if tab_selecionada == 'histogramas':
            return [dcc.Graph(figure=fig) for fig in fig_histogramas]
        elif tab_selecionada == 'boxplots':
            return [dcc.Graph(figure=fig) for fig in fig_boxplots]

    # Rodar o app
    app.run_server(debug=True)

# ----------------------------------------
# Fluxo Principal - Execução do Algoritmo
# ----------------------------------------


# 1. Carregar os dados
df, y_true = funcao_le_dados()

# 2. Gerar estatísticas descritivas
estatisticas = estatisticas_descritivas(df)

# 3. Criar gráficos descritivos (histogramas, boxplots)
fig_histogramas, fig_boxplots = criar_graficos_estatisticos(df)

# 4. Dividir os dados em treino e teste
X_train, X_test = dividir_dados(df)

# 5. Aplicar PCA aos dados de treino para reduzir para 2 dimensões
X_pca = aplicar_pca(X_train, n_components=2)

# 6. Aplicar K-means aos dados de treino
clusters_kmeans, kmeans_model = aplicar_kmeans(X_train)

# 7. Visualizar os clusters gerados pelo K-means
fig_clusters_kmeans = visualizar_clusters(X_pca, clusters_kmeans, "K-means")

# 8. Aplicar DBSCAN aos dados de treino
clusters_dbscan, dbscan_model = aplicar_dbscan(X_train)

# 9. Visualizar os clusters gerados pelo DBSCAN
fig_clusters_dbscan = visualizar_clusters(X_pca, clusters_dbscan, "DBSCAN")

# 10. Criar o Dashboard com todos os elementos
criar_dashboard(
    estatisticas,
    fig_histogramas,
    fig_boxplots,
    fig_clusters_kmeans,
    fig_clusters_dbscan
)
