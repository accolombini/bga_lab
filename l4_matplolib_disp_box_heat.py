'''
    ||> Objetivo desse exemplo é demonstrar o uso básico de três bibliotecas Pythhon para análise de dados e visualização de dados.
    
    |> Configuração SSL e Certificado
'''

# Importar bibliotecas necessárias

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ssl
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context

# Importações das Bibliotecas Necessárias

# Carregar a base de dados Iris
iris = sns.load_dataset("iris")

# Filtrar apenas as colunas numéricas para o heatmap
numeric_cols = iris.select_dtypes(include=['float', 'int'])

# Seção 1: Gráficos com Matplotlib
print("Gráficos com Matplotlib")

# Dispersão com Matplotlib
plt.figure(figsize=(8, 6))
for species in iris['species'].unique():
    subset = iris[iris['species'] == species]
    plt.scatter(subset['sepal_length'], subset['sepal_width'], label=species)
plt.title('Dispersão das Flores de Íris - Matplotlib')
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.legend(title='Espécie')
plt.show()

# Boxplot com Matplotlib
plt.figure(figsize=(8, 6))
iris.boxplot(column='petal_length', by='species')
plt.title('Boxplot do Comprimento das Pétalas - Matplotlib')
plt.suptitle('')  # Remove o título automático
plt.xlabel('Espécie')
plt.ylabel('Comprimento da Pétala')
plt.show()

# Heatmap com Matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(numeric_cols.corr(), cmap='coolwarm', interpolation='nearest')
plt.colorbar(label="Correlação")
plt.title('Mapa de Correlação - Matplotlib')
plt.xticks(range(len(numeric_cols.columns)), numeric_cols.columns, rotation=45)
plt.yticks(range(len(numeric_cols.columns)), numeric_cols.columns)
plt.show()

# Seção 2: Gráficos com Seaborn
print("Gráficos com Seaborn")

# Dispersão com Seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris, x='sepal_length', y='sepal_width',
                hue='species', style='species')
plt.title('Dispersão das Flores de Íris - Seaborn')
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.legend(title='Espécie')
plt.show()

# Boxplot com Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(data=iris, x='species', y='petal_length', palette='pastel')
plt.title('Boxplot do Comprimento das Pétalas - Seaborn')
plt.xlabel('Espécie')
plt.ylabel('Comprimento da Pétala')
plt.show()

# Heatmap com Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', square=True)
plt.title('Mapa de Correlação - Seaborn')
plt.show()

# Seção 3: Gráficos com Plotly
print("Gráficos com Plotly")

# Dispersão com Plotly
fig = px.scatter(iris, x='sepal_length', y='sepal_width', color='species',
                 title='Dispersão das Flores de Íris - Plotly')
fig.show()

# Boxplot com Plotly
fig = px.box(iris, x='species', y='petal_length',
             title='Boxplot do Comprimento das Pétalas - Plotly')
fig.show()

# Heatmap com Plotly
fig = go.Figure(data=go.Heatmap(
    z=numeric_cols.corr().values,
    x=numeric_cols.columns,
    y=numeric_cols.columns,
    colorscale='Viridis'))
fig.update_layout(title='Mapa de Correlação - Plotly')
fig.show()
