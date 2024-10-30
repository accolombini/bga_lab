'''
    ||> Estrutura do Módulo 2
        Neste módulo, nosso objetivo é criar visualizações interativas para explorar os dados e entender melhor a distribuição e a presença de outliers. Isso inclui:

        Histogramas e Boxplots Interativos: Para cada atributo do dataset, usaremos o Plotly para criar visualizações que ajudem a entender a distribuição e identificar visualmente os outliers detectados.
        Tabela de Estatísticas: Exibir estatísticas descritivas em uma tabela interativa no dashboard.
        Exibição dos Outliers: Destacar os pontos que representam outliers nas visualizações, ajudando na análise visual.
        Usaremos Dash para criar o dashboard interativo e Plotly para gerar os gráficos. Vou organizar o código para que possamos facilmente integrar tudo no final.


'''

# Segundo módulo - Visualização Exploratória e Estatística com Dash e Plotly
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from sklearn.datasets import load_iris

# Função para carregar os dados (sem a coluna target, pois é aprendizado não supervisionado)


def carregar_dados():
    iris = load_iris()
    dados = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    return dados

# Função para exibir estatísticas descritivas


def exibir_estatisticas_descritivas(dados):
    return dados.describe().reset_index()


# Carregar os dados e as estatísticas
dados = carregar_dados()
estatisticas = exibir_estatisticas_descritivas(dados)

# Inicializar o aplicativo Dash
app = dash.Dash(__name__)

# Layout do dashboard
app.layout = html.Div([
    html.H1("Análise Exploratória do Dataset Iris", style={
            'textAlign': 'center'}),  # Centralização do título principal

    # Seção de distribuição dos atributos
    html.H2("Distribuição dos Atributos", style={
            'textAlign': 'center'}),  # Centralização do subtítulo
    dcc.Tabs(id="tabs", value='tab-histogram', children=[
        dcc.Tab(label='Histogramas', value='tab-histogram'),
        dcc.Tab(label='Boxplots', value='tab-boxplot')
    ]),
    html.Div(id='tab-content', style={'display': 'flex', 'flexWrap': 'wrap'}),

    # Seção de estatísticas descritivas
    # Centralização do subtítulo
    html.H2("Estatísticas Descritivas", style={'textAlign': 'center'}),
    html.Div([
        dash_table.DataTable(
            data=estatisticas.to_dict('records'),
            columns=[{"name": i, "id": i} for i in estatisticas.columns],
            # Ajuste de largura e centralização
            style_table={'width': '60%', 'margin': '0 auto'},
            style_cell={
                'textAlign': 'center',
                'fontSize': '16px',  # Tamanho da fonte aumentado
            },
            style_header={
                'fontWeight': 'bold'
            }
        )
    ], style={'textAlign': 'center'}),
])

# Callbacks para atualizar os gráficos de acordo com a aba selecionada


@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-histogram':
        # Histogramas
        histograms = []
        for coluna in dados.columns:
            fig = px.histogram(dados, x=coluna, title=f"Histograma de {
                               coluna}", height=300, width=400)
            fig.update_layout(bargap=0.2)
            histograms.append(html.Div(dcc.Graph(figure=fig), style={
                              'display': 'inline-block', 'padding': '10px'}))
        return html.Div(histograms)

    elif tab == 'tab-boxplot':
        # Boxplots
        boxplots = []
        for coluna in dados.columns:
            fig = px.box(dados, y=coluna, title=f"Boxplot de {
                         coluna}", height=300, width=400)
            boxplots.append(html.Div(dcc.Graph(figure=fig), style={
                            'display': 'inline-block', 'padding': '10px'}))
        return html.Div(boxplots)


# Executar o aplicativo
if __name__ == '__main__':
    app.run_server(debug=True)
