'''
    Objetivo é demontra o uso de algoritmos de ML em aprendizado não supervisionado, no caso da identificação de clusters em dados de texto (flor de Iris).

    ||> Estrutura do Projeto

        |> Leitura e Carregamento dos Dados

            Carregar o dataset Iris e exibir as características no dashboard (dimensões, tipos de atributos, e estatísticas descritivas).
            Exibir a estrutura completa dos dados no dashboard, utilizando componentes de tabelas interativas do Dash.
            Estatística Descritiva e Visualização Interativa

            Calcular e exibir estatísticas descritivas no dashboard (média, mediana, desvio padrão, valores mínimos/máximos).
            Criar visualizações interativas de distribuições usando Plotly (histogramas e boxplots) para cada atributo do dataset.
            Aplicação dos Algoritmos de Agrupamento

            Aplicar o algoritmo K-means e visualizar os clusters no dashboard, utilizando gráficos de dispersão em 2D e 3D para facilitar a análise dos clusters.
            Calcular e exibir as métricas de avaliação (Silhouette Score, Calinski-Harabasz Index, e Davies-Bouldin Index) diretamente no dashboard.
            Em seguida, aplicar o algoritmo DBSCAN, repetir as visualizações e métricas para possibilitar uma comparação direta com o K-means.
            Comparação e Conclusão

            Incluir uma seção no dashboard para comparação entre os algoritmos, destacando as métricas e visualizações para facilitar a análise.
            Implementação do Projeto Usando Dash e Plotly
            Vou começar com a implementação do Dash para estruturar o dashboard e integrar os gráficos e as análises. Abaixo está o esqueleto inicial do código, que vai construir o dashboard com todas as funcionalidades descritas:
        
        ||> Ajustes e Novas Funcionalidades

            Exibição das Métricas de Análise:

            Adicionar as métricas de avaliação para cada algoritmo (Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index) diretamente no dashboard após o clique no botão.
            Apresentar as métricas em um formato claro e legível para facilitar a interpretação.
            Gráficos de Clusters:

            Incluir gráficos de dispersão 2D que mostram os clusters formados por cada algoritmo (K-means e DBSCAN) usando redução de dimensionalidade (PCA).
            Destacar visualmente os clusters com cores e legendas para facilitar a interpretação.
            Curva de Elbow para K-means:

            Adicionar um gráfico da Curva de Elbow para ajudar na escolha do número ideal de clusters, mostrando a variação da inércia (soma das distâncias ao centroide) conforme aumentamos o número de clusters.
            Comparação dos Clusters com as Classes Originais:

            Incluir uma análise detalhada no dashboard para mostrar como os clusters formados pelo K-means e DBSCAN se comparam com as classes originais do dataset (Setosa, Versicolor, Virginica).
            Exibir um gráfico de barras que mostra a distribuição real e a distribuição encontrada em cada cluster.
            Conclusão Textual:

            Adicionar um texto explicativo ao final do dashboard, resumindo os resultados e comparando os métodos, destacando qual algoritmo se aproxima mais da distribuição real.

'''

# Importando as bibliotecas necessárias

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go

# Leitura e carregamento dos dados
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
target_names = ['Setosa', 'Versicolor', 'Virginica']

# Inicializando a aplicação Dash
app = dash.Dash(__name__)

# Layout do Dashboard
app.layout = html.Div([
    html.H1("Dashboard de Clustering - Dataset Iris",
            style={'text-align': 'center'}),

    # Tabela de visualização dos dados
    html.H2("Estrutura Completa dos Dados"),
    dash_table.DataTable(
        id='datatable',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=10
    ),

    # Estatísticas descritivas
    html.H2("Estatísticas Descritivas"),
    html.Div(id='stats-output'),

    # Seção para visualizações
    html.H2("Visualizações de Distribuição"),
    dcc.Dropdown(
        id='dropdown-attribute',
        options=[{'label': col, 'value': col} for col in df.columns[:-1]],
        value='sepal length (cm)',
        clearable=False
    ),
    dcc.Graph(id='histogram'),
    dcc.Graph(id='boxplot'),

    # Curva de Elbow
    html.H2("Curva de Elbow (K-means)"),
    dcc.Graph(id='elbow-curve'),

    # Seção para agrupamento e métricas
    html.H2("Clustering e Métricas"),
    html.Button('Aplicar K-means', id='kmeans-button', n_clicks=0),
    html.Button('Aplicar DBSCAN', id='dbscan-button', n_clicks=0),
    html.Div(id='metrics-output'),
    dcc.Graph(id='scatter-plot-clusters'),

    # Tabela comparativa de clusters
    html.H2("Comparação entre Clusters Reais e Encontrados"),
    dash_table.DataTable(
        id='comparison-table',
        columns=[
            {"name": "Cluster Real", "id": "real"},
            {"name": "K-means", "id": "kmeans"},
            {"name": "DBSCAN", "id": "dbscan"}
        ],
        data=[],
        style_cell={'textAlign': 'center'},
        style_header={'fontWeight': 'bold'}
    ),

    # Conclusão
    html.H2("Conclusão", style={'font-size': '20px', 'font-weight': 'bold'}),
    html.Div(id='conclusion-output', style={'font-size': '16px'})
])

# Callbacks para atualizar as visualizações e as métricas


@app.callback(
    [Output('histogram', 'figure'),
     Output('boxplot', 'figure')],
    [Input('dropdown-attribute', 'value')]
)
def update_visualizations(attribute):
    hist_fig = px.histogram(
        df, x=attribute, title=f'Distribuição de {attribute}')
    box_fig = px.box(df, y=attribute, title=f'Boxplot de {attribute}')
    return hist_fig, box_fig


@app.callback(
    Output('stats-output', 'children'),
    Input('datatable', 'data')
)
def display_statistics(data):
    desc_stats = df.describe().reset_index()
    return dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in desc_stats.columns],
        data=desc_stats.to_dict('records')
    )


@app.callback(
    Output('elbow-curve', 'figure'),
    Input('datatable', 'data')
)
def update_elbow_curve(data):
    inertia = []
    k_values = range(1, 10)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df.iloc[:, :-1])
        inertia.append(kmeans.inertia_)

    elbow_fig = go.Figure(data=go.Scatter(
        x=list(k_values), y=inertia, mode='lines+markers'))
    elbow_fig.update_layout(title='Curva de Elbow para K-means',
                            xaxis_title='Número de Clusters (k)', yaxis_title='Inércia')
    return elbow_fig


@app.callback(
    [Output('metrics-output', 'children'),
     Output('scatter-plot-clusters', 'figure'),
     Output('comparison-table', 'data'),
     Output('conclusion-output', 'children')],
    [Input('kmeans-button', 'n_clicks'),
     Input('dbscan-button', 'n_clicks')],
    [State('comparison-table', 'data')]
)
def apply_clustering(kmeans_clicks, dbscan_clicks, existing_data):
    # Inicializa a estrutura de dados caso não exista
    if not existing_data or len(existing_data) != 3:
        existing_data = [{'real': f'{50} - {target_names[i]}',
                          'kmeans': '', 'dbscan': ''} for i in range(3)]

    if kmeans_clicks > 0:
        # Aplicando K-means
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_labels = kmeans.fit_predict(df.iloc[:, :-1])

        # Métricas para K-means
        silhouette_kmeans = silhouette_score(df.iloc[:, :-1], kmeans_labels)
        calinski_kmeans = calinski_harabasz_score(
            df.iloc[:, :-1], kmeans_labels)
        davies_kmeans = davies_bouldin_score(df.iloc[:, :-1], kmeans_labels)

        # Visualização dos clusters
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(df.iloc[:, :-1])
        scatter_fig = px.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], color=kmeans_labels,
                                 title='Clusters do K-means (PCA 2D)', labels={'color': 'Cluster'})

        # Atualiza a tabela com os resultados do K-means
        for i in range(3):
            kmeans_counts = sum(kmeans_labels == i)
            existing_data[i]['kmeans'] = f'{kmeans_counts}'

        metrics_text = f"""
        Métricas do K-means:
        - Silhouette Score: {silhouette_kmeans:.2f}
        - Calinski-Harabasz Index: {calinski_kmeans:.2f}
        - Davies-Bouldin Index: {davies_kmeans:.2f}
        """

        conclusion_text = """
        O K-means conseguiu identificar clusters que, em grande parte, correspondem às classes originais (Setosa, Versicolor, Virginica).
        No entanto, é importante observar a distribuição e a sobreposição entre as classes.
        """

        return metrics_text, scatter_fig, existing_data, conclusion_text

    elif dbscan_clicks > 0:
        # Aplicando DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(df.iloc[:, :-1])

        # Métricas para DBSCAN
        silhouette_dbscan = silhouette_score(
            df.iloc[:, :-1], dbscan_labels) if len(set(dbscan_labels)) > 1 else 'N/A'
        calinski_dbscan = calinski_harabasz_score(
            df.iloc[:, :-1], dbscan_labels) if len(set(dbscan_labels)) > 1 else 'N/A'
        davies_dbscan = davies_bouldin_score(
            df.iloc[:, :-1], dbscan_labels) if len(set(dbscan_labels)) > 1 else 'N/A'

        # Visualização dos clusters
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(df.iloc[:, :-1])
        scatter_fig = px.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], color=dbscan_labels,
                                 title='Clusters do DBSCAN (PCA 2D)', labels={'color': 'Cluster'})

        # Atualiza a tabela com os resultados do DBSCAN
        for i in range(3):
            dbscan_counts = sum((df['target'] == i) & (dbscan_labels != -1))
            existing_data[i]['dbscan'] = f'{dbscan_counts}'

        metrics_text = f"""
        Métricas do DBSCAN:
        - Silhouette Score: {silhouette_dbscan}
        - Calinski-Harabasz Index: {calinski_dbscan}
        - Davies-Bouldin Index: {davies_dbscan}
        """

        conclusion_text = """
        O DBSCAN identificou clusters com base na densidade, o que pode resultar em uma segmentação diferente em relação ao K-means.
        Isso pode ser útil para dados com formas mais complexas, mas é importante ajustar os parâmetros para obter bons resultados.
        """

        return metrics_text, scatter_fig, existing_data, conclusion_text

    return "", {}, existing_data, ""


# Executando o servidor Dash
if __name__ == '__main__':
    app.run_server(debug=True)
