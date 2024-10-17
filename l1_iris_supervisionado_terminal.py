
'''
    Objetivo: demonstrar como é o comportamento de uma aplicação ML Supervisionada, no caso na presença de um dataset com um classificador, definindo a categoria das flores de iris.
    Também vamos usar dois algoritmos de aprendizado supervisionado: KNN (K-Nearest Neighbors) e SVM (Support Vector Machines).
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
# Dividir o dataset em 70% treino e 30% teste, garantindo estratificação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Função para reordenar a matriz de confusão com base na ordem Setosa, Versicolor, Virginica
def reordenar_matriz(matriz):
    '''
    Reorganiza a matriz de confusão para garantir que as classes 
    estejam na ordem correta: Setosa, Versicolor, Virginica.
    '''
    # Índices correspondentes a Setosa (0), Versicolor (1), Virginica (2)
    ordem = [0, 1, 2]
    # Reorganiza as linhas e colunas da matriz de confusão
    matriz_reordenada = matriz[np.ix_(ordem, ordem)]
    return matriz_reordenada

# Treinando e avaliando os modelos KNN e SVM
def avaliar_modelos():
    # Inicializando os modelos
    knn = KNeighborsClassifier(n_neighbors=3)  # k definido como 3 como exemplo
    svm = SVC(kernel='linear', random_state=42)

    # Treinando o modelo KNN
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    # Treinando o modelo SVM
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    # Avaliando e reordenando a matriz de confusão para ambos os modelos
    matriz_knn = confusion_matrix(y_test, y_pred_knn)
    matriz_svm = confusion_matrix(y_test, y_pred_svm)

    print("Matriz de Confusão KNN (reordenada):")
    print(reordenar_matriz(matriz_knn))
    print("\nRelatório de Classificação KNN:")
    print(classification_report(y_test, y_pred_knn))

    print("\nMatriz de Confusão SVM (reordenada):")
    print(reordenar_matriz(matriz_svm))
    print("\nRelatório de Classificação SVM:")
    print(classification_report(y_test, y_pred_svm))

# Chamar a função para avaliar os modelos
avaliar_modelos()
