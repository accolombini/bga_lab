'''
    Objetivo: vamos trabalhar com a base Wine Quality por falta de uma base decente para trabalhar com dados relacionados à engenharia elétrica.

    Ações:
        FP-Tree: Usaremos uma estrutura de árvore de padrões frequentes (FP-Tree), semelhante ao algoritmo FP-Growth, para encontrar padrões frequentes nos dados.
        Classificação com Múltiplas Regras de Associação: Após identificar os padrões frequentes, o CMAR utiliza essas regras para classificação, escolhendo as regras mais relevantes para prever uma classe alvo.
        Proposta para Implementação
        Passo 1: Encontrar uma base de dados relacionada à engenharia elétrica.

        Uma sugestão é utilizar dados sobre falhas de transformadores, qualidade de energia elétrica, ou consumo de energia residencial/industrial, disponíveis em repositórios como o UCI Machine Learning Repository ou Kaggle.
        Passo 2: Implementação do CMAR

        Dividir o dataset em transações com classes associadas para treinamento.
        Construir uma FP-Tree para encontrar os itemsets frequentes.
        Gerar múltiplas regras de associação a partir dos itemsets frequentes e usá-las para classificar novas instâncias.

    Implementação do Algoritmo CMAR
        Estrutura do Algoritmo CMAR
        Pré-processamento dos Dados:

        Carregar e explorar os dados.
        Transformar as características em transações, onde cada instância é uma transação e a classe de falha é o rótulo.
        Construção da FP-Tree e Geração de Itemsets Frequentes:

        Usaremos uma estrutura de árvore de padrões frequentes (FP-Tree) para encontrar itemsets frequentes, similar ao FP-Growth.
        Geração de Múltiplas Regras de Associação para Classificação:

        Após identificar os itemsets frequentes, geramos regras de associação e escolhemos as mais relevantes para classificar novas instâncias.
        Classificação Baseada em Múltiplas Regras de Associação:

        Usamos as regras para prever a classe de novas instâncias, baseando-se nas regras mais significativas.

    ||> Conclusão: O algoritmo CMAR, embora potente em dados categóricos e discretos com padrões de associação, enfrenta limitações nos dados contínuos e dispersos da base Wine Quality. A recomendação é migrar para métodos como Random Forest, que são mais adequados para capturar as complexidades dos dados contínuos e oferecerão uma performance mais consistente. Teste o código Random Forest para observar melhorias significativas na precisão e balanceamento das previsões entre as classes "high" e "low".
'''

# Importar as bibliotecas necessárias
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mlxtend.preprocessing import TransactionEncoder

# Carregar o dataset Wine Quality
# Este dataset contém características físico-químicas de vinhos e uma coluna de qualidade
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Transformar a coluna de qualidade em classes binárias: "high" (>=7) e "low" (<7)
# Facilita a classificação ao dividir o vinho em duas categorias de qualidade
data['quality_class'] = data['quality'].apply(
    lambda x: 'high' if x >= 7 else 'low')
data = data.drop(columns=['quality'])  # Remover a coluna original de qualidade

# Dividir dados em conjuntos de treino e teste
# A proporção de 70% para treino e 30% para teste ajuda a avaliar o modelo
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['quality_class']),
                                                    data['quality_class'],
                                                    test_size=0.3, random_state=42)

# Criar transações formatadas para o FP-Growth
# Cada linha do conjunto de treino é transformada em uma transação com os valores das características
transactions = []
for _, row in X_train.iterrows():
    transaction = [f"{col}={int(val)}" for col,
                   val in row.items()]  # Ex.: "pH=3"
    transactions.append(transaction)

# Transformar transações em um DataFrame binário para aplicar FP-Growth
# TransactionEncoder cria uma matriz binária onde cada coluna é uma característica e cada linha uma transação
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Aplicar FP-Growth para encontrar itemsets frequentes
# min_support é ajustado para um valor baixo para capturar itemsets mais raros
min_support = 0.001
frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

# Gerar regras de associação com base nos itemsets frequentes
# min_confidence baixo para capturar mais regras e observar mais associações
min_confidence = 0.1
rules = association_rules(
    frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Separar regras específicas para cada classe ("high" e "low")
# Filtra as regras onde o consequente é "quality_class=high" ou "quality_class=low"
rules_high = rules[rules['consequents'].apply(
    lambda x: 'quality_class=high' in x)]
rules_low = rules[rules['consequents'].apply(
    lambda x: 'quality_class=low' in x)]

# Concatenar as regras para as duas classes e garantir que ambas tenham representatividade
classification_rules = pd.concat([rules_high, rules_low])

# Verificar se alguma regra de classificação foi gerada
if classification_rules.empty:
    print("Nenhuma regra de classificação foi gerada.")
else:
    print(f"Total de regras de classificação geradas: {
          len(classification_rules)}")

# Função para classificar uma instância do conjunto de teste com base nas regras de associação geradas


def classify_instance(instance, rules):
    # Seleciona as regras cujos antecedentes estão contidos na instância
    relevant_rules = rules[rules['antecedents'].apply(
        lambda x: x.issubset(instance))]
    if not relevant_rules.empty:
        # Usa votação ponderada para determinar a classe com base na confiança das regras
        class_votes = relevant_rules.groupby('consequents')['confidence'].sum()
        return class_votes.idxmax()[0]  # Retorna a classe com maior confiança
    # Retorna "high" como fallback se nenhuma regra aplicável for encontrada
    return 'high'


# Aplicar o classificador ao conjunto de teste
# Para cada instância no conjunto de teste, cria uma transação e prevê a classe
predictions = []
for _, row in X_test.iterrows():
    # Converte cada linha em um conjunto de características
    instance = set([f"{col}={int(val)}" for col, val in row.items()])
    prediction = classify_instance(instance, classification_rules)
    predictions.append(prediction)

# Avaliar o desempenho do classificador
# Gera uma avaliação de precisão, recall e f1-score para as classes "high" e "low"
print(classification_report(y_test, predictions, labels=['high', 'low']))
