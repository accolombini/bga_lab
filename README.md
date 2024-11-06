## Projetos de aplicação de ML para Big Analytcs

#### O ambiente virtual para esses laboratórios é o: bd_a

**Importante:**  construir um exemplo de uma aplicação usando aprendizado supervisionado, quero explorar 
todoas as funções estatísticas (estatística descritiva), utilizar a biblioteca plotly e dash para os 
diagramas, precisamos além das funções estatísticas gerar a matriz de confusão e tirar todas as conclusões 
desta aplicação. As saídas todas devem ser direcionadas para um Dashboard gerado com as bibliotecas acima.

*__Nota:__* explorar os comentários e estes devem ser bem auto explicativos, por exemplo # Dividir o conjunto 
de dados em treino e teste, ser espcífico # Dividir o conjunto de dados de treinamento e testes na proporção 
70% 30%. Ao carregar o conjunto de dados mostrar as caracterísitcas dos mesmos, por exemplo qual a 
dimensão da base, quais são os atributos e destacar o atributo classificador. 
Por fim trabalhar com funções, por exemplo, Def funcao_le_dados, e assim por diante.

>É desejo que toda saída seja direcionada para um Dashboard construído usando Plotly e Dash.

>Neste projeto, vamos incluir todas as aplicações desenvolvidas para os alunos
>Notadamente, vamos trabalhar com ML não supervisionada

- Flor de Iris - Supervisionada
- Flor de Iris - Não Supervisionada

#### Estrutura do projeto:

### Leitura e carregamento dos dados:

>Carregar o conjunto de dados "Flor de Iris".
Mostrar as características do conjunto de dados (dimensão, tipos de atributos, distribuição).
Destacar o atributo classificador.

### Estatística descritiva:

>Calcular e exibir métricas estatísticas como média, mediana, desvio padrão, etc.
Gerar gráficos (histogramas, boxplots, etc.) para visualizar a distribuição dos dados.

### Divisão dos dados:

>Separar o conjunto de dados em 70% para treino e 30% para teste.
Treinamento do modelo:

Escolher um algoritmo de aprendizado supervisionado (por exemplo, KNN e  SVM).
Treinar o modelo e fazer previsões sobre os dados de teste.

### Matriz de confusão e métricas de avaliação:

>Gerar uma matriz de confusão e calcular métricas como precisão, recall e F1-score.

### Criação do Dashboard:

>Usar Dash e Plotly para criar um Dashboard interativo que mostre os dados, gráficos estatísticos e resultados do modelo (como a matriz de confusão).

### Proposta de Funções:
>def funcao_le_dados(): Para carregar e exibir informações iniciais do dataset.
def funcao_estatisticas_descritivas(): Para gerar as estatísticas descritivas dos atributos.
def funcao_visualizacoes(): Para gerar as visualizações gráficas com Plotly.
def funcao_dividir_treino_teste(): Para dividir os dados em treino e teste e mostrar as distribuições.
def funcao_treinar_knn(): Para treinar o modelo KNN e exibir os resultados.
def funcao_treinar_svm(): Para treinar o modelo SVM e exibir os resultados.
def funcao_matriz_confusao(): Para gerar a matriz de confusão e comparar o desempenho dos modelos.
