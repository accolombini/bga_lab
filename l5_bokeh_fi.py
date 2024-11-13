'''
    Objetivo: trabalhar um pouco com a biblioteca bokeh para visualizar dados.

    Visão Geral e Interatividade:

        Este código cria um gráfico de dispersão que permite ao usuário escolher quais variáveis visualizar no eixo X e no eixo Y (ex.: comprimento da sépala, largura da pétala).
        Os widgets Select (menus suspensos) permitem a seleção interativa das variáveis para análise, o que torna a experiência dinâmica e envolvente.
        Personalização:

        A visualização utiliza um esquema de cores e uma transparência (alpha=0.5) que realçam as diferenças entre as espécies.
        A legenda é configurada com um título "Espécies" e uma política de clique que permite ocultar uma espécie ao clicar sobre o nome dela na legenda.
        Configurações como o tamanho dos círculos e as ferramentas de zoom e reset tornam a visualização mais refinada e personalizável.
        Ferramentas Interativas:

        A visualização inclui ferramentas como pan, wheel_zoom, box_zoom, reset e hover. A ferramenta hover exibe informações sobre o ponto ao passar o mouse sobre ele, o que enriquece a experiência de exploração dos dados.
        Streaming de Dados:

        Embora o exemplo não inclua dados em tempo real, o ColumnDataSource do Bokeh permite que dados sejam atualizados dinamicamente. Com uma fonte de dados que envia atualizações, é possível adicionar streaming de dados ao gráfico.
        Execução em Ambiente de Aplicações Web:

        Esse código usa curdoc().add_root(layout) para que possa ser executado como uma aplicação Bokeh em um servidor (via bokeh serve), permitindo uma experiência totalmente interativa e independente.

        Pan (Mover) pan
        Descrição: Permite "arrastar" o gráfico para movê-lo em qualquer direção.
        Uso: Clique e segure o botão esquerdo do mouse, arraste o gráfico na direção desejada.

        2. Wheel Zoom (Zoom com Roda do Mouse) wheel_zoom
        Descrição: Permite ampliar ou reduzir o zoom usando a roda do mouse.
        Uso: Role a roda do mouse para aproximar ou afastar a visualização.

        3. Box Zoom (Zoom por Seleção de Área) box_zoom
        Descrição: Permite aplicar o zoom a uma área específica do gráfico selecionando-a com o mouse.
        Uso: Clique e arraste para desenhar uma caixa em torno da área que você deseja ampliar.

        4. Reset (Reiniciar) reset
        Descrição: Restaura o gráfico para a visualização original, removendo qualquer ajuste de zoom ou movimento feito anteriormente.
        Uso: Clique uma vez para redefinir o gráfico ao estado inicial.

        5. Hover (Dica de Ferramenta) hover
        Descrição: Exibe informações detalhadas sobre o ponto ou elemento sobre o qual o cursor está passando.
        Uso: Passe o mouse sobre um ponto no gráfico para ver dados como coordenadas e outros atributos configurados para exibição.

        6. Save (Salvar como Imagem) save
        Descrição: Permite salvar a visualização como uma imagem em formato PNG.
        Uso: Clique para baixar uma captura de tela do gráfico atual.

        7. Box Select (Seleção Retangular) box_select
        Descrição: Permite selecionar pontos específicos no gráfico desenhando uma caixa ao redor deles. Útil para selecionar subconjuntos de dados.
        Uso: Clique e arraste para desenhar uma caixa de seleção ao redor dos pontos desejados.

        8. Lasso Select (Seleção com Laço) lasso_select
        Descrição: Semelhante ao Box Select, mas permite selecionar pontos desenhando uma forma livre ao redor deles.
        Uso: Clique e desenhe ao redor dos pontos que você deseja selecionar.

        9. Crosshair (Mira) crosshair
        Descrição: Exibe uma linha de mira horizontal e vertical que segue o cursor, facilitando a localização exata de pontos em gráficos.
        Uso: Movimente o cursor pelo gráfico para visualizar a posição da mira.

        10. Tap (Seleção por Clique) tap
        Descrição: Permite selecionar pontos individuais ao clicar sobre eles, útil em visualizações que suportam interatividade adicional, como detalhes de um ponto específico.
        Uso: Clique em um ponto no gráfico para selecioná-lo.

        11. Poly Select (Seleção Poligonal) poly_select
        Descrição: Permite selecionar pontos desenhando uma área poligonal. Isso é útil para fazer seleções personalizadas que seguem formas mais complexas.
        Uso: Clique e desenhe ao redor dos pontos que deseja selecionar; feche o polígono para completar a seleção.

        12. Zoom In e Zoom Out (Aproximar e Afastar) zoom_in, zoom_out
        Descrição: Ferramentas de zoom fixo que permitem ampliar ou reduzir a visualização em incrementos específicos.
        Uso: Clique para aplicar zoom in ou zoom out no gráfico.

        ||> Nota: Algumas dessas ferramentas, como lasso_select e poly_select, são mais úteis para gráficos que permitem seleções e filtragens interativas. Dependendo do contexto do seu gráfico, você pode escolher quais ferramentas são mais apropriadas para oferecer uma experiência de visualização ideal.

        ||> Importante -> para executar esse código, você precisa ter o Bokeh instalado em seu ambiente. Se você não o tem, você pode instalá-lo usando pip: pip install bokeh. E rodar o servidor bokeh serve no seu terminal: bokeh serve. => bokeh serve --show l5_bokeh_fi.py

'''

# Importar bibliotecas necessárias
import pandas as pd  # Manipulação de dados
from bokeh.plotting import figure  # Para criação de gráficos no Bokeh
# Fonte de dados e widgets interativos
from bokeh.models import ColumnDataSource, Select
from bokeh.layouts import column  # Organização do layout
from bokeh.io import curdoc  # Integração com o servidor Bokeh
import requests  # Para download de dados via HTTP
from io import StringIO  # Manipulação de dados no formato de string como arquivos

# Carregar o dataset Iris diretamente da web
# URL do dataset Iris no repositório da UCI Machine Learning
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
response = requests.get(url)  # Realizar a requisição HTTP para obter os dados
# Carregar o dataset em um DataFrame com nomes de colunas especificados
data = pd.read_csv(StringIO(response.text), header=None, names=[
                   "sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

# Criar uma fonte de dados para o gráfico, usando o DataFrame carregado
source = ColumnDataSource(data)

# Função para criar a figura de dispersão usando 'scatter'


def create_figure(x_axis, y_axis):
    # Definir a figura e configurações básicas
    # Configura título, rótulos dos eixos e ferramentas de interação
    p = figure(title="Gráfico Interativo de Iris",
               x_axis_label=x_axis,
               y_axis_label=y_axis,
               width=700,
               height=400,
               tools="pan,wheel_zoom,box_zoom,reset,hover,save")

    # Adicionar os dados de dispersão ao gráfico usando 'scatter'
    # Representa cada espécie de flor com pontos personalizados
    p.scatter(x=x_axis, y=y_axis, source=source, size=8,
              color="navy", alpha=0.5, legend_field="species")

    # Personalizações adicionais
    # Adiciona título e posição da legenda, permitindo ocultar itens ao clicar
    p.legend.title = "Espécies"
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"  # Permite esconder espécies na legenda ao clicar

    return p  # Retorna a figura criada


# Widgets interativos para seleção dos eixos
# Dropdown para seleção da variável do eixo X
x_select = Select(title="Eixo X", value="sepal_length", options=[
                  "sepal_length", "sepal_width", "petal_length", "petal_width"])
# Dropdown para seleção da variável do eixo Y
y_select = Select(title="Eixo Y", value="sepal_width", options=[
                  "sepal_length", "sepal_width", "petal_length", "petal_width"])

# Criar a figura inicial com os valores padrão dos seletores
p = create_figure(x_select.value, y_select.value)

# Função de callback para atualizar o gráfico quando os seletores são alterados


def update_plot(attr, old, new):
    # Atualizar o gráfico com os novos valores dos eixos escolhidos
    new_figure = create_figure(x_select.value, y_select.value)
    layout.children[2] = new_figure  # Substitui o gráfico existente no layout


# Associar o callback Python aos seletores
# Configura o callback para que o gráfico seja atualizado ao mudar o eixo X ou Y
x_select.on_change("value", update_plot)
y_select.on_change("value", update_plot)

# Organizar o layout da aplicação com os widgets e o gráfico
layout = column(x_select, y_select, p)

# Exibir a aplicação Bokeh usando o servidor Bokeh
# Adiciona o layout ao documento atual, permitindo que o servidor Bokeh o exiba
curdoc().add_root(layout)
