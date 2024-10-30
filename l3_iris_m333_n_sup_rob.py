'''
    ||> Estrutura do Módulo 3: Clustering e Algoritmos Avançados
        Neste módulo, vamos focar em:

        Curva Elbow para Determinação do Número de Clusters:

        Implementaremos a curva Elbow para sugerir o número ideal de clusters.
        Exibir no dashboard para auxiliar na visualização da escolha.
        Algoritmos Avançados de Clustering:

        t-SNE e UMAP: Aplicaremos esses métodos de redução de dimensionalidade e visualizaremos os clusters em 2D.
        Autoencoders: Utilizaremos Autoencoders para criar uma representação latente dos dados.
        GANs e Transformers (opcional): Exploraremos essas técnicas para observar a separação dos dados e os padrões de agrupamento.
        Vou iniciar com a Curva Elbow e os métodos de t-SNE e UMAP, para que possamos iterar e validar essa parte antes de avançar para os algoritmos mais avançados.

    ||> Passo a Passo
        Curva Elbow:

        Vamos calcular a Inércia (Soma das Distâncias Quadráticas Internas) para diferentes números de clusters.
        Exibiremos um gráfico interativo com a curva Elbow no dashboard.
        Redução de Dimensionalidade com t-SNE e UMAP:

        Aplicaremos t-SNE e UMAP para reduzir as dimensões do dataset Iris para duas dimensões e exibir a separação dos clusters.
        Essas visualizações ajudarão a observar os agrupamentos sugeridos pelo algoritmo.
'''

# Terceiro módulo - Curva Elbow e Visualização com t-SNE (ajuste para títulos acima dos gráficos)
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Função para carregar e padronizar os dados


def carregar_dados():
    iris = load_iris()
    dados = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(dados)
    return dados_padronizados, dados

# Função para calcular a curva Elbow


def calcular_curva_elbow(dados, max_clusters=10):
    inercia = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(dados)
        inercia.append(kmeans.inertia_)
    return inercia

# Função para redução de dimensionalidade com t-SNE e aplicação de KMeans para colorir clusters


def aplicar_tsne_e_kmeans(dados, n_clusters=3):
    # Redução de dimensionalidade com t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    dados_tsne = tsne.fit_transform(dados)
    tsne_df = pd.DataFrame(dados_tsne, columns=[
                           'Componente 1', 'Componente 2'])

    # Aplicação do KMeans para rotular clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    tsne_df['Cluster'] = kmeans.fit_predict(dados)
    return tsne_df


# Carregar os dados e calcular a curva Elbow
dados_padronizados, dados_originais = carregar_dados()
curva_elbow = calcular_curva_elbow(dados_padronizados)

# Aplicar t-SNE e KMeans para visualizar os clusters
dados_tsne = aplicar_tsne_e_kmeans(dados_padronizados, n_clusters=3)

# Inicializar o aplicativo Dash
app = dash.Dash(__name__)

# Layout do dashboard
app.layout = html.Div([
    html.H1("Análise de Clustering - Curva Elbow e t-SNE",
            style={'textAlign': 'center', 'fontSize': '32px'}),

    # Curva Elbow
    html.Div([
        html.H2("Curva Elbow para Determinação do Número de Clusters",
                style={'textAlign': 'center', 'fontSize': '24px'}),
        html.Div(
            dcc.Graph(
                id='curva-elbow',
                figure=px.line(x=list(range(1, 11)), y=curva_elbow,
                               markers=True, title="Curva Elbow", height=400, width=600)
                # Centralização do título
                .update_layout(xaxis_title="Número de Clusters", yaxis_title="Inércia", title_x=0.5)
                # Centralização do gráfico
            ), style={'display': 'flex', 'justifyContent': 'center'}
        )
    ], style={'padding': '20px'}),

    # Visualização de t-SNE com clusters coloridos
    html.Div([
        html.H2("Visualização de Clusters com t-SNE",
                style={'textAlign': 'center', 'fontSize': '24px'}),
        html.Div(
            dcc.Graph(
                id='tsne-plot',
                figure=px.scatter(dados_tsne, x='Componente 1', y='Componente 2', color='Cluster', title="Clusters com t-SNE",
                                  height=400, width=600)
                .update_traces(marker=dict(size=5))
                .update_layout(title_x=0.5)  # Centralização do título
                # Centralização do gráfico
            ), style={'display': 'flex', 'justifyContent': 'center'}
        )
    ], style={'padding': '20px'}),
])

# Executar o aplicativo
if __name__ == '__main__':
    app.run_server(debug=True)
