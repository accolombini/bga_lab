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

    |||> Como alternativa vamos usar o Random Forest como alternativa ao CMAR, pois ele é mais adequado para capturar as complexidades dos dados contínuos e oferecerão uma boa visão do que temos nesta base.
'''


# Importar bibliotecas necessárias

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.tree import export_text

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

# Criar e treinar o modelo Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Fazer previsões
predictions = clf.predict(X_test)

# Avaliar o desempenho do classificador
print(classification_report(y_test, predictions, labels=['high', 'low']))

# Mostrar Importâncias das Características
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': clf.feature_importances_
}).sort_values(by='importance', ascending=False)

print("Importâncias das Características:\n", feature_importances)

# Exibir regras de uma árvore individual no Random Forest
# Escolher uma árvore para exemplificar as regras
tree_sample = clf.estimators_[0]
tree_rules = export_text(tree_sample, feature_names=list(X_train.columns))
print("\nAmostra de Regras de uma Árvore do Random Forest:\n", tree_rules)
