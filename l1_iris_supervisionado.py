'''
    Objetivo: demonstrar como é o comportamento de uma aplicação ML Supervisionada, no caso na presença de um dataset com um classificador, definindo a categoria das flores de iris.
    Também vmos usar dois algoritmos de aprendizado supervisionado: KNN (K-Nearest Neighbors) e SVM (Support Vector Machines).
'''

# Importar bibliotecas necessárias
import dash  # Biblioteca para criar o dashboard
from dash import dcc, html  # Componentes principais de layout e gráficos
from dash import dash_table  # Tabela dinâmica para o dashboard
import plotly.figure_factory as ff  # Para gerar gráficos, incluindo heatmaps
from sklearn.datasets import load_iris  # Carregar o dataset Iris
# Dividir os dados em treino e teste
from sklearn.model_selection import train_test_split
# Métricas para avaliar os modelos
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier  # Classificador KNN
from sklearn.svm import SVC  # Classificador SVM
import pandas as pd  # Manipulação de dados em DataFrame
# Manipulação numérica (para reordenar a matriz de confusão)
import numpy as np

# Inicializando o objeto principal da aplicação Dash
app = dash.Dash(__name__)

# Carregar o dataset Iris e organizar em um DataFrame pandas
iris = load_iris()  # Carregar os dados
# Criar um DataFrame com as features do dataset e adicionar a coluna 'classe' para as classes
dados = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dados['classe'] = iris.target  # Adicionar a coluna de classes (alvo)

# Divisão dos dados em treino e teste
X = dados.iloc[:, :-1]  # Todas as colunas exceto a última (features)
y = dados['classe']  # Coluna de classes (alvo)
# Dividir o dataset em 70% treino e 30% teste, garantindo randomização
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Função para reordenar a matriz de confusão com base na ordem Setosa, Versicolor, Virginica


def reordenar_matriz(matriz):
    """
    Reorganiza a matriz de confusão para garantir que as classes 
    estejam na ordem correta: Setosa, Versicolor, Virginica.
    """
    # Índices correspondentes a Setosa (0), Versicolor (1), Virginica (2)
    ordem = [0, 1, 2]
    # Reorganiza as linhas e colunas da matriz de confusão
    matriz_reordenada = matriz[np.ix_(ordem, ordem)]
    return matriz_reordenada

# Função para gerar a matriz de confusão com rótulos e valores corrigidos


def gerar_heatmap(matriz):
    """
    Gera o heatmap (matriz de confusão) com os rótulos corrigidos e os valores organizados.
    O eixo X representa as predições e o eixo Y representa os valores reais.
    """
    # Rótulos das classes para os eixos
    x = ['Setosa', 'Versicolor', 'Virginica']  # Predições
    y = ['Setosa', 'Versicolor', 'Virginica']  # Valores reais

    # Criar o heatmap com rótulos e cores apropriadas
    fig = ff.create_annotated_heatmap(
        matriz[::-1], x=x, y=y[::-1], colorscale='Viridis', showscale=True)

    # Configurar o layout do gráfico (títulos e dimensões)
    fig.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Predito",
        yaxis_title="Real",
        width=400,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)  # Tamanho do gráfico
    )

    return fig

# Funções de treinamento para os modelos

# Treinamento do modelo KNN


def treinar_knn():
    """
    Treina o modelo KNN (K-Nearest Neighbors) e retorna a matriz de confusão corrigida
    e o relatório de classificação.
    """
    knn = KNeighborsClassifier(
        n_neighbors=5)  # Instanciar o modelo KNN com 5 vizinhos
    knn.fit(X_train, y_train)  # Treinar o modelo com os dados de treino

    y_pred = knn.predict(X_test)  # Fazer predições com os dados de teste

    # Gerar a matriz de confusão e reordená-la para a ordem correta
    matriz_knn = confusion_matrix(y_test, y_pred)
    matriz_knn_corrigida = reordenar_matriz(matriz_knn)

    # Retornar a matriz corrigida e o relatório de classificação
    return matriz_knn_corrigida, classification_report(y_test, y_pred, output_dict=True)

# Treinamento do modelo SVM


def treinar_svm():
    """
    Treina o modelo SVM (Support Vector Machine) e retorna a matriz de confusão corrigida
    e o relatório de classificação.
    """
    svm = SVC(kernel='linear', C=0.1)  # Instanciar o modelo SVM com kernel linear
    svm.fit(X_train, y_train)  # Treinar o modelo com os dados de treino

    y_pred = svm.predict(X_test)  # Fazer predições com os dados de teste

    # Gerar a matriz de confusão e reordená-la para a ordem correta
    matriz_svm = confusion_matrix(y_test, y_pred)
    matriz_svm_corrigida = reordenar_matriz(matriz_svm)

    # Retornar a matriz corrigida e o relatório de classificação
    return matriz_svm_corrigida, classification_report(y_test, y_pred, output_dict=True)


# Resultados dos modelos
matriz_knn, relatorio_knn = treinar_knn()  # Treinar e obter resultados do KNN
matriz_svm, relatorio_svm = treinar_svm()  # Treinar e obter resultados do SVM

# Função para gerar a conclusão


def gerar_conclusao(relatorio, modelo_nome):
    """
    Gera uma conclusão baseada nos resultados do relatório de classificação de cada modelo.
    """
    precisao = relatorio['weighted avg']['precision']  # Precisão do modelo
    recall = relatorio['weighted avg']['recall']  # Recall do modelo
    f1_score = relatorio['weighted avg']['f1-score']  # F1-score do modelo

    # Retorna a string com a conclusão formatada
    return (f"O modelo {modelo_nome} apresentou uma precisão de {precisao:.2f}, "
            f"recall de {recall:.2f} e F1-score de {f1_score:.2f}.")


# Layout do dashboard
app.layout = html.Div([
    # Título principal do dashboard
    html.H1("Análise de Modelos Supervisionados: KNN e SVM",
            style={'textAlign': 'center', 'fontSize': '28px'}),

    # Seção de estatísticas descritivas dos atributos
    html.H3("Estatísticas Descritivas dos Atributos", style={
            'textAlign': 'center', 'fontSize': '24px'}),
    dash_table.DataTable(
        id='tabela-descritiva',
        columns=[{"name": i, "id": i}
                 for i in dados.describe().reset_index().columns],
        data=dados.describe().reset_index().to_dict('records'),
        style_table={'width': '70%', 'margin': 'auto'},
    ),

    # Divisão do conjunto de dados em treino e teste
    html.H3("Divisão do Conjunto de Dados em Treino e Teste", style={
            'textAlign': 'center', 'fontSize': '24px'}),
    html.Div([
        html.P(f"Número de amostras no conjunto de treino: {len(X_train)}",
               style={'fontSize': '20px'}),
        html.P(f"Número de amostras no conjunto de teste: {len(X_test)}",
               style={'fontSize': '20px'}),
    ], style={'textAlign': 'center'}),

    # Seção de resultados do KNN
    html.H3("Resultados do KNN", style={
            'textAlign': 'center', 'fontSize': '24px'}),
    html.Div(dcc.Graph(id='matriz-knn', figure=gerar_heatmap(matriz_knn)),
             style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'}),
    dash_table.DataTable(
        id='tabela-knn',
        columns=[{"name": i, "id": i}
                 for i in ['precision', 'recall', 'f1-score', 'support']],
        data=pd.DataFrame(relatorio_knn).transpose().to_dict('records'),
        style_table={'width': '50%', 'margin': 'auto'}
    ),

    # Seção de resultados do SVM
    html.H3("Resultados do SVM", style={
            'textAlign': 'center', 'fontSize': '24px'}),
    html.Div(dcc.Graph(id='matriz-svm', figure=gerar_heatmap(matriz_svm)),
             style={'textAlign': 'center', 'display': 'flex', 'justify-content': 'center'}),
    dash_table.DataTable(
        id='tabela-svm',
        columns=[{"name": i, "id": i}
                 for i in ['precision', 'recall', 'f1-score', 'support']],
        data=pd.DataFrame(relatorio_svm).transpose().to_dict('records'),
        style_table={'width': '50%', 'margin': 'auto'}
    ),

    # Seção de conclusões
    html.H3("Conclusão", style={'textAlign': 'center', 'fontSize': '24px'}),
    html.Div(id='conclusao', children=[
        html.P(gerar_conclusao(relatorio_knn, "KNN"),
               style={'fontSize': '20px'}),
        html.P(gerar_conclusao(relatorio_svm, "SVM"),
               style={'fontSize': '20px'}),
    ], style={'textAlign': 'center', 'width': '70%', 'margin': 'auto'})
])

# Executar a aplicação Dash
if __name__ == '__main__':
    app.run_server(debug=True)
