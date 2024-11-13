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

# Importar bibliotecas necessárias

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mlxtend.preprocessing import TransactionEncoder

# Carregar o dataset Wine Quality
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Transformar a coluna de qualidade em classes binárias: "high" (>=7) e "low" (<7)
data['quality_class'] = data['quality'].apply(
    lambda x: 'high' if x >= 7 else 'low')
data = data.drop(columns=['quality'])

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data.drop(
    columns=['quality_class']), data['quality_class'], test_size=0.3, random_state=42)

# Criar transações formatadas para a FP-Growth
transactions = []
for _, row in X_train.iterrows():
    transaction = [f"{col}={int(val)}" for col, val in row.items()]
    transactions.append(transaction)

# Transformar transações em DataFrame binário
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Aplicar FP-Growth para encontrar itemsets frequentes com parâmetros mínimos
min_support = 0.001  # Mais baixo possível para capturar todas as regras
frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

# Gerar regras de associação com base nos itemsets frequentes
min_confidence = 0.1  # Baixo para capturar mais regras
rules = association_rules(
    frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Separar regras para ambas as classes
rules_high = rules[rules['consequents'].apply(
    lambda x: 'quality_class=high' in x)]
rules_low = rules[rules['consequents'].apply(
    lambda x: 'quality_class=low' in x)]

# Garantir que ambas as classes tenham regras aplicáveis
classification_rules = pd.concat([rules_high, rules_low])

if classification_rules.empty:
    print("Nenhuma regra de classificação foi gerada.")
else:
    print(f"Total de regras de classificação geradas: {
          len(classification_rules)}")

# Função para classificar uma instância com fallback na classe "high" se não houver regra aplicável


def classify_instance(instance, rules):
    relevant_rules = rules[rules['antecedents'].apply(
        lambda x: x.issubset(instance))]
    if not relevant_rules.empty:
        # Votação ponderada para determinar a classe com base nas regras
        class_votes = relevant_rules.groupby('consequents')['confidence'].sum()
        return class_votes.idxmax()[0]  # Classe com maior soma de confiança
    # Forçar previsão para "high" como fallback para capturar mais casos de "high"
    return 'high'


# Aplicar o classificador CMAR ao conjunto de teste
predictions = []
for _, row in X_test.iterrows():
    instance = set([f"{col}={int(val)}" for col, val in row.items()])
    prediction = classify_instance(instance, classification_rules)
    predictions.append(prediction)

# Avaliar o desempenho do classificador
print(classification_report(y_test, predictions, labels=['high', 'low']))
