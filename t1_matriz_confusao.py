'''
    Objetivo aquei é construir um script Python para testar a matriz de confusão.
'''
# Importar as bibliotecas necessárias

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import plotly.figure_factory as ff

# Carregar o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Treinar um modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Gerar a matriz de confusão (corrigindo os rótulos e valores)
matriz_confusao = confusion_matrix(y_test, y_pred)

# Função para gerar o heatmap da matriz de confusão


def gerar_heatmap(matriz):
    # Aqui estão os rótulos corretos tanto para X (predito) quanto Y (real)
    x = ['Setosa', 'Versicolor', 'Virginica']  # Predições
    y = ['Setosa', 'Versicolor', 'Virginica']  # Valores reais

    fig = ff.create_annotated_heatmap(
        matriz[::-1], x=x, y=y[::-1], colorscale='Viridis', showscale=True)

    # Ajuste do layout
    fig.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Predito",
        yaxis_title="Real",
        width=400,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


# Gerar a figura da matriz de confusão
figura_matriz = gerar_heatmap(matriz_confusao)
figura_matriz.show()
