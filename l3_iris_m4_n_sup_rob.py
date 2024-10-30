'''
    ||> Estrutura do Último Módulo
        Autoencoders para Redução de Dimensionalidade:

        Vamos usar um autoencoder para reduzir a dimensionalidade do dataset Iris a um espaço latente de duas dimensões.
        Isso nos permitirá observar como os clusters se distribuem no espaço latente criado pela rede neural.
        Visualização do Espaço Latente:

        Exibir o resultado do autoencoder em um gráfico de dispersão, similar ao que fizemos com o t-SNE.
        Análise Comparativa e Conclusão:

        Fornecer uma breve análise sobre a eficácia dos métodos aplicados (t-SNE e autoencoder).
        Discutir qual método se mostrou mais eficaz na separação dos clusters e a adequação para o dataset Iris.

    ||> Estrutura Alternativa: t-SNE com GANs
        Já usamos o t-SNE para uma visualização preliminar, mas agora podemos utilizar um GAN para tentar criar uma representação mais robusta dos clusters no espaço latente. A ideia seria:

        Treinar um GAN para gerar pontos de dados similares ao dataset Iris. Em vez de um autoencoder, o GAN vai gerar novos exemplos com características próximas dos dados originais.
        Aplicar t-SNE no espaço gerado pelo GAN para observar se conseguimos identificar melhor os clusters.
        Essa abordagem nos permite observar padrões e estrutura dos dados com uma técnica de geração de dados.

    ||> Para avaliar a qualidade dos clusters em aprendizado não supervisionado, embora não tenhamos rótulos como em aprendizado supervisionado, existem métricas intrínsecas que podem nos ajudar a entender a coesão e a separação dos clusters. Vamos explorar algumas delas que são aplicáveis em nosso caso:

        |>1. Índice de Silhueta (Silhouette Score)
        O Índice de Silhueta mede o quão próximo cada ponto de um cluster está dos pontos do mesmo cluster (coesão) em relação aos pontos dos clusters vizinhos (separação). Ele varia de -1 a 1:

        Valores próximos de 1 indicam clusters bem separados.
        Valores próximos de 0 indicam que os clusters estão sobrepostos.
        Valores negativos indicam que os pontos estão próximos de clusters incorretos.
        |> 2. Índice de Davies-Bouldin
        Essa métrica calcula a média da similaridade entre cada cluster e o cluster mais semelhante a ele. Valores menores indicam melhores separações entre clusters.

        |>3. Índice de Calinski-Harabasz (CH Score)
        O Índice de Calinski-Harabasz mede a dispersão entre clusters em comparação com a dispersão dentro dos clusters. Quanto maior o valor, melhor a separação entre os clusters.

        ||> Implementação da Métrica de Silhueta
            Vou adicionar a métrica de Silhueta ao código para que possamos avaliar a qualidade dos clusters gerados. Vamos calcular o índice de Silhueta para os dados originais e os dados sintéticos separadamente.
    
        |>1. Definições: Original vs Sintético
            Dados Originais: São os dados reais do conjunto de dados Iris, que contém 150 amostras com três classes (Setosa, Versicolor, Virginica), que sabemos que devem formar três clusters naturais.
            Dados Sintéticos: São dados gerados pelo modelo GAN que tentam imitar a distribuição dos dados originais. A ideia é que, ao final do treinamento, esses dados sintéticos reflitam uma estrutura de clusters semelhante à dos dados originais.
        |> 2. Coeficiente de Silhueta: Interpretação
            O coeficiente de Silhueta varia entre -1 e 1:

            Valores próximos de 1: Indicam que os pontos estão bem agrupados em seus respectivos clusters e estão distantes de outros clusters, o que é um bom sinal de separação.
            Valores próximos de 0: Indicam que os pontos estão próximos das fronteiras dos clusters, sugerindo sobreposição entre os clusters.
            Valores negativos: Indicam que alguns pontos podem estar agrupados com o cluster errado.
            Resultados no Dashboard
            Índice de Silhueta - Dados Originais: 0.59: Esse valor indica uma separação razoável, mas não ideal, entre os clusters nos dados originais. É esperado que os dados do Iris tenham uma estrutura de clusters mais clara, mas a aplicação do t-SNE, que é uma técnica de redução de dimensionalidade, pode estar comprimindo essa separação.

            Índice de Silhueta - Dados Sintéticos: 0.71: Esse valor é um pouco mais alto, sugerindo que os clusters nos dados sintéticos estão um pouco mais definidos. Isso pode ocorrer porque o GAN conseguiu capturar uma estrutura de cluster aproximada, mas com menos variabilidade ou sobreposição do que os dados originais.

        |>3. Conclusão
            Os resultados indicam que:

            O modelo GAN conseguiu criar uma estrutura de cluster que reflete aproximadamente a dos dados originais. O índice de Silhueta dos dados sintéticos é ligeiramente superior, sugerindo que eles estão "mais separados" do que os clusters nos dados reais. Isso pode ocorrer porque o GAN gera dados de uma forma que aproxima a distribuição, mas com menos nuances ou variações sutis.
            No entanto, nenhum dos índices de Silhueta está próximo de 1, o que sugere que ainda há sobreposição entre clusters, especialmente nos dados originais, conforme visualizado na dispersão.
            Esses insights podem ser usados para aprimorar o modelo ou ajustar os parâmetros do GAN para tentar uma geração mais precisa, ou para investigar se há outras técnicas de visualização ou clusterização que possam revelar ainda melhor a estrutura dos dados.

    -----------------------------
        1. O Índice de Silhueta e a Separação dos Clusters
        O índice de Silhueta mede principalmente a separação dos clusters (o quão bem eles estão afastados uns dos outros). Um valor maior de Silhueta nos dados sintéticos (0.59) indica que esses clusters estão relativamente bem separados. No entanto, isso não significa necessariamente que os clusters estejam formados de maneira precisa ou que reflitam a estrutura real dos dados originais.

        Por outro lado, nos dados originais (com índice de Silhueta de 0.55), os clusters podem estar mais sobrepostos ou ter mais variabilidade interna, o que leva a um índice de Silhueta ligeiramente mais baixo. Contudo, essa variabilidade é, na verdade, uma representação mais fiel da complexidade dos dados reais, onde as classes podem ter características ligeiramente mais sobrepostas.

        2. Dados Sintéticos e a Natureza do GAN
        Os dados sintéticos gerados pelo GAN são projetados para capturar a distribuição dos dados originais, mas com uma simplificação ou "suavização" dessa distribuição. Isso ocorre porque:

        O GAN tenta reproduzir a distribuição geral dos dados, mas, sem um sinal direto supervisionado, ele pode introduzir uma certa "rigidez" ou "simplificação" nos clusters.
        Essa simplificação faz com que os clusters sintéticos pareçam mais bem separados do que realmente são nos dados originais. Isso se reflete em um índice de Silhueta mais alto, mas não em uma contagem de clusters mais precisa.
        3. Contagem de Clusters: Por que os Dados Originais São Mais Precisos
        A contagem de clusters nos dados originais é mais precisa porque os dados reais possuem as características detalhadas e as variações sutis presentes nas flores de Iris, refletindo a distribuição natural das classes. Essas variações permitem que o algoritmo de clusterização (KMeans) identifique com mais precisão os três clusters distintos.

        Nos dados sintéticos, como o GAN simplifica a distribuição (para capturar a "essência" dos dados), ele pode não gerar pontos com a mesma variação interna dos dados reais. Como resultado:

        O KMeans pode acabar agrupando alguns pontos sintéticos no cluster errado.
        A contagem de clusters para os dados sintéticos se torna menos precisa, pois os clusters gerados podem não corresponder exatamente às fronteiras dos clusters reais.
        Conclusão
        Portanto:

        Índice de Silhueta mais alto nos sintéticos: Os clusters sintéticos são mais bem separados, mas isso é um reflexo da simplificação da distribuição.
        Contagem mais precisa nos dados originais: Os dados reais têm uma variação mais complexa, que reflete melhor as classes verdadeiras, permitindo uma contagem de clusters mais fiel.
        Esse comportamento é um exemplo clássico de como métricas não supervisionadas como o índice de Silhueta podem oferecer uma visão, mas não capturam todos os aspectos da precisão dos clusters em relação aos dados originais.
    
'''

# Módulo final com métricas de Silhueta e visualização de clusters com t-SNE
import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Função para carregar e padronizar os dados


def carregar_dados():
    iris = load_iris()
    dados = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(dados)
    return dados_padronizados

# Definir a arquitetura do GAN


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Treinamento do GAN


def treinar_gan(dados, epochs=10000):
    input_dim = dados.shape[1]
    generator = Generator(input_dim=input_dim, output_dim=input_dim)
    discriminator = Discriminator(input_dim=input_dim)

    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        real_data = torch.tensor(dados, dtype=torch.float32)
        real_labels = torch.ones((dados.shape[0], 1))
        fake_labels = torch.zeros((dados.shape[0], 1))

        # Treinamento do Discriminador
        optimizer_d.zero_grad()
        outputs = discriminator(real_data)
        loss_real = criterion(outputs, real_labels)

        noise = torch.randn(dados.shape[0], input_dim)
        fake_data = generator(noise)
        outputs = discriminator(fake_data.detach())
        loss_fake = criterion(outputs, fake_labels)

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # Treinamento do Gerador
        optimizer_g.zero_grad()
        outputs = discriminator(fake_data)
        loss_g = criterion(outputs, real_labels)
        loss_g.backward()
        optimizer_g.step()

    return generator

# Geração de dados sintéticos usando o GAN


def gerar_dados_sinteticos(generator, num_samples):
    input_dim = generator.model[0].in_features
    noise = torch.randn(num_samples, input_dim)
    dados_sinteticos = generator(noise).detach().numpy()
    return dados_sinteticos

# Aplicação do t-SNE e K-means para visualizar clusters nos dados originais e sintéticos


def aplicar_tsne_kmeans(dados_originais, dados_sinteticos, n_clusters=3):
    tsne = TSNE(n_components=2, random_state=0)
    dados_concatenados = np.vstack((dados_originais, dados_sinteticos))
    dados_tsne = tsne.fit_transform(dados_concatenados)

    # K-means para rotular clusters nos dados concatenados
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(dados_concatenados)

    # Criar dataframe para visualização
    df_tsne = pd.DataFrame(dados_tsne, columns=[
                           'Componente 1', 'Componente 2'])
    df_tsne['Tipo'] = ['Original'] * \
        len(dados_originais) + ['Sintético'] * len(dados_sinteticos)
    df_tsne['Cluster'] = labels
    return df_tsne

# Calcular a métrica de Silhueta


def calcular_silhouette(df_tsne):
    # Filtrar dados originais e sintéticos
    dados_originais = df_tsne[df_tsne['Tipo'] == 'Original']
    dados_sinteticos = df_tsne[df_tsne['Tipo'] == 'Sintético']

    # Calcular o Silhouette Score para dados originais e sintéticos
    score_original = silhouette_score(
        dados_originais[['Componente 1', 'Componente 2']], dados_originais['Cluster'])
    score_sintetico = silhouette_score(
        dados_sinteticos[['Componente 1', 'Componente 2']], dados_sinteticos['Cluster'])

    return score_original, score_sintetico


# Carregar dados, treinar GAN e gerar dados sintéticos
dados_originais = carregar_dados()
gan = treinar_gan(dados_originais)
dados_sinteticos = gerar_dados_sinteticos(
    gan, num_samples=dados_originais.shape[0])

# Aplicar t-SNE e K-means nos dados originais e sintéticos
dados_tsne = aplicar_tsne_kmeans(dados_originais, dados_sinteticos)

# Cálculo das métricas de Silhueta para dados originais e sintéticos
score_original, score_sintetico = calcular_silhouette(dados_tsne)

# Gerar tabela explicativa dos clusters


def gerar_tabela_clusters(df_tsne):
    tabela = df_tsne.groupby(
        ['Tipo', 'Cluster']).size().reset_index(name='Contagem')
    return tabela


# Dados para a tabela
tabela_clusters = gerar_tabela_clusters(dados_tsne)

# Inicializar o aplicativo Dash
app = dash.Dash(__name__)

# Layout do dashboard
app.layout = html.Div([
    html.H1("GANs e t-SNE - Visualização de Clusters com Métricas de Avaliação",
            style={'textAlign': 'center', 'fontSize': '32px'}),

    # Exibição das métricas de Silhueta
    html.Div([
        html.H2(f"Índice de Silhueta - Dados Originais: {score_original:.2f}", style={
                'textAlign': 'center', 'fontSize': '20px'}),
        html.H2(f"Índice de Silhueta - Dados Sintéticos: {
                score_sintetico:.2f}", style={'textAlign': 'center', 'fontSize': '20px'}),
    ], style={'padding': '10px'}),

    # Visualização t-SNE dos dados originais e sintéticos com clusters
    html.Div([
        html.H2("Clusters com t-SNE (Dados Originais vs Sintéticos)",
                style={'textAlign': 'center', 'fontSize': '24px'}),
        html.Div(
            dcc.Graph(
                id='gan-tsne-plot',
                figure=px.scatter(dados_tsne, x='Componente 1', y='Componente 2', color='Cluster',
                                  symbol='Tipo', title="Clusters com t-SNE - Dados Originais e Sintéticos",
                                  height=500, width=700,
                                  color_continuous_scale=px.colors.qualitative.Bold)
                .update_traces(marker=dict(size=5))
                .update_layout(title_x=0.5, legend=dict(yanchor="top", y=1, xanchor="right", x=1))
            ), style={'display': 'flex', 'justifyContent': 'center'}
        )
    ], style={'padding': '20px'}),

    # Tabela de explicação dos clusters
    html.H2("Tabela de Clusters - Dados Originais e Sintéticos",
            style={'textAlign': 'center', 'fontSize': '24px'}),
    html.Div([
        dash_table.DataTable(
            data=tabela_clusters.to_dict('records'),
            columns=[{"name": i, "id": i} for i in tabela_clusters.columns],
            style_table={'width': '60%', 'margin': '0 auto'},
            style_cell={'textAlign': 'center', 'fontSize': '16px'},
            style_header={'fontWeight': 'bold'}
        )
    ], style={'padding': '20px'}),
])

# Executar o aplicativo
if __name__ == '__main__':
    app.run_server(debug=True)
