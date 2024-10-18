'''
    Objetivo é demontrar o uso de algoritmos de ML em aprendizado não supervisionado, no caso da identificação de clusters em dados de texto (flor de Iris).

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

            Uso do PCA: O uso do PCA é recomendado para projetar os dados em 2D/3D para visualizações. Embora os dados do Iris tenham apenas 4 dimensões, a redução de dimensionalidade para 2 ou 3 componentes facilita a visualização e não impacta negativamente o clustering. Logo, o uso do PCA aqui é mais para fins visuais, e não estritamente necessário para o agrupamento em si.

            Curva de Elbow para K-means: Adicionar a curva do método do cotovelo (Elbow) para o K-means ajuda a escolher o número ótimo de clusters. Isso é bastante comum e recomendado.
            4. Comparação e Conclusão
            Comparação entre K-means e DBSCAN: Comparar os clusters formados com as classes reais do dataset (Setosa, Versicolor, Virginica) é uma excelente ideia. Usar um gráfico de barras para mostrar a correspondência entre os clusters e as classes reais facilita a interpretação.
            Conclusão textual: Incluir um resumo comparando os dois métodos e destacando qual se aproxima mais da distribuição real é uma ótima abordagem para concluir o trabalho.
            Dúvida sobre o uso do PCA
            O PCA não é estritamente necessário para os algoritmos de clustering, pois tanto o K-means quanto o DBSCAN podem operar nos dados originais sem problemas. No entanto, o PCA é muito útil para visualizações dos clusters, especialmente quando queremos projetar dados de alta dimensionalidade em um espaço 2D ou 3D. Como você planeja criar gráficos de dispersão para visualização, o uso de PCA é apropriado para reduzir os dados para duas ou três dimensões.

            ||> Resumo sobre o PCA:

            Necessário?: Não, os algoritmos de clustering podem ser aplicados diretamente aos dados originais.
            Recomendado?: Sim, para visualizações 2D e 3D no dashboard, o PCA facilita a interpretação dos clusters de forma visual.

        |||> Resultados:
            1. Silhouette Score
                Intervalo: [−1,1]
                Significado: Essa métrica avalia a separação entre os clusters. Um valor próximo de 1 indica que os clusters estão bem separados e cada ponto está mais próximo do seu próprio cluster do que de qualquer outro. Valores negativos indicam que os pontos estão mais próximos de clusters diferentes, sugerindo má separação.
                
                |> Resultados:

                    K-means: 0.5512
                    DBSCAN: 0.4860
                    Interpretação: O K-means obteve um Silhouette Score mais alto, o que sugere que seus clusters estão melhor definidos e mais separados entre si, enquanto o DBSCAN apresenta uma separação um pouco inferior. Ambos têm valores acima de 0.4, o que ainda indica uma qualidade de clustering razoável, mas com o K-means sendo superior neste aspecto.

            2. Calinski-Harabasz Index
                Intervalo: Quanto maior, melhor.
                Significado: Essa métrica mede a relação entre a dispersão dentro dos clusters e a dispersão entre os clusters. Valores maiores indicam uma maior separação entre os clusters e uma melhor compactação dos pontos dentro de seus próprios clusters.
                
                |> Resultados:
                    K-means: 561.5937
                    DBSCAN: 220.2975
                    Interpretação: O índice Calinski-Harabasz é muito maior para o K-means, sugerindo que os clusters formados por ele são mais compactos e melhor separados entre si, enquanto o DBSCAN obteve um valor bem mais baixo, indicando que os clusters formados por ele são mais difusos e menos bem definidos.

            3. Davies-Bouldin Index
                Intervalo: Quanto menor, melhor.
                Significado: Mede a proximidade entre os clusters. Valores menores indicam que os clusters estão bem separados uns dos outros. Valores maiores indicam que os clusters estão mais próximos entre si ou que há sobreposição.
                
                |> Resultados:
                    K-means: 0.6660
                    DBSCAN: 7.2224
                    Interpretação: O índice Davies-Bouldin de K-means é muito menor do que o de DBSCAN. Isso sugere que os clusters formados pelo K-means estão muito mais bem separados entre si, enquanto no DBSCAN, os clusters estão mais sobrepostos ou mal definidos.

            ||> Interpretação Geral:
                    Com base nessas três métricas, podemos concluir que o K-means está fornecendo clusters de melhor qualidade comparado ao DBSCAN para o dataset Iris:

                    Silhouette Score do K-means é ligeiramente superior, sugerindo uma separação melhor entre os clusters.
                    Calinski-Harabasz Index é significativamente maior para o K-means, indicando que os clusters são mais compactos e bem separados.
                    Davies-Bouldin Index é muito menor para o K-means, reforçando que os clusters estão bem distantes uns dos outros, enquanto no DBSCAN, há uma maior sobreposição entre clusters.
                    Sobre o DBSCAN:
                    O DBSCAN pode estar tendo dificuldades para lidar com os dados do dataset Iris. Como o Iris é um conjunto de dados com clusters mais compactos e esféricos, o K-means tende a funcionar melhor. O DBSCAN, por outro lado, é mais adequado para dados com clusters de forma irregular ou com ruídos, e parece que ele não está encontrando uma boa separação nos dados do Iris.

            |||>O que pode ser feito:
                    Ajustar eps e min_samples: Tentar diferentes valores para esses parâmetros, especialmente reduzindo eps para ver se o DBSCAN consegue identificar clusters mais coerentes.
                    Considerar o uso do K-means: Para o dataset Iris, o K-means parece estar fornecendo resultados mais coerentes com a distribuição real dos dados.

'''

# Importar bibliotecas necessárias para a aplicação
from sklearn.decomposition import PCA  # Para redução de dimensionalidade
from sklearn.cluster import KMeans, DBSCAN  # Algoritmos de clustering
# Métricas de avaliação
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px  # Biblioteca para visualizações interativas
import dash  # Dash para criar o dashboard
from dash import dcc, html, dash_table  # Componentes do Dash
import pandas as pd  # Manipulação de dados
from sklearn import datasets  # Para carregar o dataset Iris
import plotly.graph_objects as go  # Para gráficos comparativos
import numpy as np  # Biblioteca para operações numéricas

# 1. Carregando o dataset Iris
# O dataset Iris contém 4 características das flores e suas respectivas classes (Setosa, Versicolor, Virginica)
iris = datasets.load_iris()
# Criando um DataFrame Pandas para organizar os dados
df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
df['target'] = iris['target']  # Adicionando a coluna de classes
# Mapeando os valores para nomes de classe
df['target_name'] = df['target'].map(
    {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# 2. Configurando o aplicativo Dash
app = dash.Dash(__name__)

# Função para comparar os clusters formados pelo K-means ou DBSCAN com as classes reais


def compare_clusters(clusters, method_name):
    """
    Compara os clusters gerados com as classes reais do dataset Iris
    utilizando uma tabela de contingência e gera um gráfico de barras.

    Parâmetros:
    clusters (array): clusters formados pelo algoritmo.
    method_name (str): nome do algoritmo (K-means ou DBSCAN).

    Retorna:
    fig (go.Figure): gráfico de barras comparativo.
    """
    # Criando uma tabela de contingência entre clusters e classes reais
    comparison = pd.crosstab(df['target_name'], clusters, rownames=[
                             'Classe Real'], colnames=[f'Clusters ({method_name})'])

    # Criando o gráfico de barras da comparação
    fig = go.Figure()

    for class_name in comparison.index:
        fig.add_trace(go.Bar(
            x=comparison.columns,
            y=comparison.loc[class_name],
            name=class_name
        ))

    # Configurando o layout do gráfico
    fig.update_layout(
        title=f"Comparação entre Clusters ({method_name}) e Classes Reais",
        barmode='stack',  # Empilhar as barras para facilitar a visualização
        xaxis_title="Clusters",
        yaxis_title="Quantidade de Amostras",
        legend_title="Classe Real"
    )

    return fig


# 3. Layout do Dashboard
# Definindo a estrutura da interface do usuário com componentes interativos
app.layout = html.Div([
    html.H1("Análise de Clustering no Dataset Iris",
            style={'fontSize': 32, 'textAlign': 'center'}),

    # Tabela interativa para visualização dos dados do dataset Iris
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i}
                 for i in df.columns],  # Colunas da tabela
        data=df.to_dict('records'),  # Dados em formato de dicionário
        page_size=10,  # Número de registros por página
        style_header={'fontSize': 20},  # Tamanho da fonte do cabeçalho
        style_data={'fontSize': 18},  # Tamanho da fonte dos dados
        style_cell={'textAlign': 'center'}  # Alinhamento do conteúdo
    ),

    # Div para exibir as estatísticas descritivas de cada atributo
    html.Div(id='stats', style={'fontSize': 18}),

    # Dropdown para o usuário selecionar o atributo para análise
    html.Label('Selecione um atributo para visualização:',
               style={'fontSize': 18}),
    dcc.Dropdown(
        id='dropdown-feature',  # Identificador do dropdown
        # Opções para seleção (atributos do dataset)
        options=[{'label': col, 'value': col} for col in df.columns[:-2]],
        value=df.columns[0],  # Valor inicial
        style={'fontSize': 18, 'width': '50%'}
    ),

    # Gráficos para exibir histogramas e boxplots interativos com base no atributo selecionado
    dcc.Graph(id='histogram'),  # Gráfico do histograma
    dcc.Graph(id='boxplot'),  # Gráfico do boxplot

    # Seção para o algoritmo K-means
    html.H2("K-means", style={'fontSize': 24}),
    html.Label('Selecione o número de clusters para K-means:',
               style={'fontSize': 18}),
    dcc.Input(id='num-clusters', type='number', value=3,
              min=2, max=10, step=1, style={'fontSize': 18}),
    # Gráfico de dispersão 2D para visualização dos clusters do K-means
    dcc.Graph(id='kmeans-2d-scatter'),
    # Div para exibir as métricas do K-means
    html.Div(id='kmeans-metrics', style={'fontSize': 18}),

    # Seção para a curva de Elbow do K-means
    html.H2("Curva de Elbow", style={'fontSize': 24}),
    html.Label('Número máximo de clusters para a curva de Elbow:',
               style={'fontSize': 18}),
    dcc.Input(id='max-clusters', type='number', value=10,
              min=2, max=20, step=1, style={'fontSize': 18}),
    dcc.Graph(id='elbow-curve'),  # Gráfico para exibir a curva de Elbow

    # Seção para o algoritmo DBSCAN
    html.H2("DBSCAN", style={'fontSize': 24}),
    html.Label('Defina o valor de epsilon (EPS):', style={'fontSize': 18}),
    dcc.Input(id='eps', type='number', value=0.5, min=0.1,
              max=2.0, step=0.1, style={'fontSize': 18}),
    html.Label('Defina o número mínimo de amostras:', style={'fontSize': 18}),
    dcc.Input(id='min_samples', type='number', value=5,
              min=1, max=10, step=1, style={'fontSize': 18}),
    # Gráfico de dispersão 2D para visualização dos clusters do DBSCAN
    dcc.Graph(id='dbscan-2d-scatter'),
    # Div para exibir as métricas do DBSCAN
    html.Div(id='dbscan-metrics', style={'fontSize': 18}),

    # Seção de comparação entre clusters e classes reais
    html.H2("Comparação entre Clusters e Classes Reais",
            style={'fontSize': 24}),
    # Gráfico de barras para comparação com K-means
    dcc.Graph(id='comparison-kmeans'),
    # Gráfico de barras para comparação com DBSCAN
    dcc.Graph(id='comparison-dbscan'),

    # Seção de conclusão
    html.H2("Conclusão Comparativa", style={'fontSize': 24}),
    html.Div(id='conclusion', style={'fontSize': 18})
])


# 4. Função callback para exibir as estatísticas descritivas com base no atributo selecionado
@app.callback(
    dash.dependencies.Output('stats', 'children'),
    [dash.dependencies.Input('dropdown-feature', 'value')]
)
def display_statistics(selected_feature):
    """
    Calcula e exibe as estatísticas descritivas do atributo selecionado
    pelo usuário a partir do dropdown.

    Parâmetro:
    selected_feature (str): nome do atributo selecionado.

    Retorna:
    html.Div: Div com as estatísticas descritivas.
    """
    # Verifica se o atributo selecionado é válido
    if selected_feature is None or selected_feature not in df.columns:
        return html.Div("Selecione um atributo válido.")

    # Calcula as estatísticas descritivas
    stats = df[selected_feature].describe()
    return html.Div([
        html.H4(f"Estatísticas descritivas para {selected_feature}"),
        html.P(f"Contagem: {stats['count']}"),
        html.P(f"Média: {stats['mean']:.2f}"),
        html.P(f"Mediana: {df[selected_feature].median():.2f}"),
        html.P(f"Desvio Padrão: {stats['std']:.2f}"),
        html.P(f"Valor Mínimo: {stats['min']:.2f}"),
        html.P(f"Valor Máximo: {stats['max']:.2f}")
    ])


# 5. Função callback para gerar os gráficos de histograma e boxplot interativos
@app.callback(
    [dash.dependencies.Output('histogram', 'figure'),
     dash.dependencies.Output('boxplot', 'figure')],
    [dash.dependencies.Input('dropdown-feature', 'value')]
)
def update_graphs(selected_feature):
    """
    Atualiza os gráficos de histograma e boxplot com base no
    atributo selecionado pelo usuário.

    Parâmetro:
    selected_feature (str): nome do atributo selecionado.

    Retorna:
    histogram (dict): dicionário representando o gráfico de histograma.
    boxplot (dict): dicionário representando o gráfico de boxplot.
    """
    if selected_feature is None or selected_feature not in df.columns:
        return {}, {}

    # Histograma do atributo selecionado
    histogram = {
        'data': [{'x': df[selected_feature], 'type': 'histogram'}],
        'layout': {'title': f"Histograma de {selected_feature}"}
    }

    # Boxplot do atributo selecionado
    boxplot = {
        'data': [{'y': df[selected_feature], 'type': 'box'}],
        'layout': {'title': f"Boxplot de {selected_feature}"}
    }

    return histogram, boxplot


# 6. Função callback para rodar o K-means e exibir os clusters e as métricas
@app.callback(
    [dash.dependencies.Output('kmeans-2d-scatter', 'figure'),
     dash.dependencies.Output('kmeans-metrics', 'children')],
    [dash.dependencies.Input('num-clusters', 'value')]
)
def apply_kmeans(num_clusters):
    """
    Executa o algoritmo K-means no dataset Iris e gera um gráfico
    de dispersão 2D dos clusters formados e as métricas de avaliação.

    Parâmetro:
    num_clusters (int): número de clusters selecionado pelo usuário.

    Retorna:
    fig (dict): gráfico de dispersão 2D dos clusters.
    metrics (list): lista de métricas de avaliação (Silhouette, Calinski-Harabasz, Davies-Bouldin).
    """
    # Verificando se o número de clusters é válido
    if num_clusters is None or num_clusters < 1:
        return {}, []

    try:
        # Aplicando K-means ao dataset
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        clusters = kmeans.fit_predict(df[iris['feature_names']])

        # Redução de dimensionalidade com PCA para visualização em 2D
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df[iris['feature_names']])

        # Verificando se o PCA retornou valores válidos
        if df_pca.shape[1] < 2:
            return {}, []

        # Criando o gráfico de dispersão 2D dos clusters
        fig = px.scatter(x=df_pca[:, 0], y=df_pca[:, 1],
                         color=clusters, title="Clusters K-means (2D)")

        # Calculando as métricas de avaliação
        silhouette = silhouette_score(df[iris['feature_names']], clusters)
        calinski_harabasz = calinski_harabasz_score(
            df[iris['feature_names']], clusters)
        davies_bouldin = davies_bouldin_score(
            df[iris['feature_names']], clusters)

        # Exibindo as métricas
        metrics = [
            html.P(f"Silhouette Score: {silhouette:.4f}"),
            html.P(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}"),
            html.P(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        ]

        return fig, metrics

    except Exception as e:
        # Retorna um gráfico vazio e nenhuma métrica se houver um erro
        return {}, [html.P(f"Erro ao aplicar K-means: {str(e)}")]


# 7. Função callback para gerar a curva de Elbow do K-means

@app.callback(
    dash.dependencies.Output('elbow-curve', 'figure'),
    [dash.dependencies.Input('max-clusters', 'value')]
)
def plot_elbow_curve(max_clusters):
    distortions = []
    K = list(range(1, max_clusters + 1))

    # Verifica se o valor de max_clusters é válido
    if max_clusters is None or max_clusters < 2:
        return {}

    # Calcula a inércia (distortion) para diferentes números de clusters
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df[iris['feature_names']])
        distortions.append(kmeans.inertia_)

    # Verifica se há dados suficientes para plotar o gráfico
    if len(distortions) > 1:
        # Gera o gráfico de Elbow com os valores calculados
        fig = px.line(x=K, y=distortions, title="Curva de Elbow")
        fig.update_layout(xaxis_title="Número de Clusters",
                          yaxis_title="Inércia (Distortion)")
    else:
        # Retorna um gráfico vazio se os dados não forem suficientes
        fig = {}

    return fig


# 8. Função callback para rodar o DBSCAN e exibir os clusters e as métricas


# Função callback para rodar o DBSCAN e exibir os clusters e as métricas
@app.callback(
    [dash.dependencies.Output('dbscan-2d-scatter', 'figure'),
     dash.dependencies.Output('dbscan-metrics', 'children')],
    [dash.dependencies.Input('eps', 'value'),
     dash.dependencies.Input('min_samples', 'value')]
)
def apply_dbscan(eps, min_samples):
    # Verificação de parâmetros válidos
    if eps is None or min_samples is None or eps <= 0 or min_samples <= 0:
        return {}, [html.P("Parâmetros de EPS ou min_samples inválidos.")]

    try:
        # Aplicando DBSCAN ao dataset com parâmetros fornecidos
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(df[iris['feature_names']])

        # Redução de dimensionalidade com PCA para visualização em 2D
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df[iris['feature_names']])

        # Verificando se o DBSCAN retornou valores válidos
        if len(set(clusters)) <= 1:
            return {}, [html.P("DBSCAN não conseguiu identificar clusters suficientes. Tente ajustar EPS ou min_samples.")]

        # Criando o gráfico de dispersão 2D dos clusters
        fig = px.scatter(x=df_pca[:, 0], y=df_pca[:, 1],
                         color=clusters, title="Clusters DBSCAN (2D)")

        # Calculando as métricas de avaliação se houver mais de um cluster
        if len(set(clusters)) > 1:
            silhouette = silhouette_score(df[iris['feature_names']], clusters)
            calinski_harabasz = calinski_harabasz_score(
                df[iris['feature_names']], clusters)
            davies_bouldin = davies_bouldin_score(
                df[iris['feature_names']], clusters)

            metrics = [
                html.P(f"Silhouette Score: {silhouette:.4f}"),
                html.P(f"Calinski-Harabasz Index: {calinski_harabasz:.4f}"),
                html.P(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
            ]
        else:
            metrics = [
                html.P("Não foi possível calcular as métricas com apenas 1 cluster.")]

        return fig, metrics

    except Exception as e:
        return {}, [html.P(f"Erro ao aplicar DBSCAN: {str(e)}")]


# 9. Função callback para gerar a comparação entre K-means e as classes reais


@app.callback(
    dash.dependencies.Output('comparison-kmeans', 'figure'),
    [dash.dependencies.Input('num-clusters', 'value')]
)
def compare_kmeans(num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(df[iris['feature_names']])
    return compare_clusters(clusters, "K-means")

# Função callback para gerar a comparação entre DBSCAN e as classes reais


@app.callback(
    dash.dependencies.Output('comparison-dbscan', 'figure'),
    [dash.dependencies.Input('eps', 'value'),
     dash.dependencies.Input('min_samples', 'value')]
)
def compare_dbscan(eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df[iris['feature_names']])
    return compare_clusters(clusters, "DBSCAN")

# Função callback para gerar a conclusão comparativa


@app.callback(
    dash.dependencies.Output('conclusion', 'children'),
    [dash.dependencies.Input('num-clusters', 'value'),
     dash.dependencies.Input('eps', 'value'),
     dash.dependencies.Input('min_samples', 'value')]
)
def generate_conclusion(num_clusters, eps, min_samples):
    # Calculando o K-means e DBSCAN
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters_kmeans = kmeans.fit_predict(df[iris['feature_names']])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters_dbscan = dbscan.fit_predict(df[iris['feature_names']])

    # Contagem de quantos clusters foram formados
    num_clusters_kmeans = len(np.unique(clusters_kmeans))
    num_clusters_dbscan = len(np.unique(clusters_dbscan))

    # Comparando as métricas e gerando uma conclusão
    conclusion = [
        html.P(f"O algoritmo K-means formou {num_clusters_kmeans} clusters."),
        html.P(f"O algoritmo DBSCAN formou {num_clusters_dbscan} clusters."),
        html.P("Com base nas métricas de avaliação (Silhouette, Calinski-Harabasz, Davies-Bouldin) e na visualização, "
               "é possível observar que o K-means tende a formar clusters mais definidos em relação às classes reais, "
               "enquanto o DBSCAN pode ser mais eficiente em detectar estruturas mais complexas de clusters.")
    ]

    return conclusion


# 10. Rodando o aplicativo Dash
if __name__ == '__main__':
    app.run_server(debug=True)
