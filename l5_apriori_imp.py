'''
    Objetivo: passos para criar um script utilizando o algoritmo Apriori.
    Para estes exemplo utilizaremos o conjunto de dados públiocos Online Retail Dataset (Online Retail) do UCI Machine Learning Repository. Para efeito de demonstração (tempo de execução, vamos usar uma versão reduzida, chamada Dataset de Transações de Exemplo)
'''

# Importar bibliotecas necessárias
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Dados de exemplo
dataset = [
    ['Leite', 'Ovos'],
    ['Leite', 'Biscoitos'],
    ['Leite', 'Ovos', 'Biscoitos'],
    ['Ovos', 'Biscoitos']
]

# Transformar dados
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Aplicar Apriori
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print("Itemsets Frequentes (Itens mais comprados juntos):\n")
print("Suporte indica a proporção de transações que contêm o itemset. Quanto maior o suporte, mais frequente é o itemset.")
print(frequent_itemsets)

# Gerar e exibir as regras de associação
min_confidence = 0.5
rules = association_rules(
    frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Exibir as regras de associação, se existirem
if not rules.empty:
    print("\nRegras de Associação (Relações encontradas entre os itens):\n")
    print("As regras mostram a confiança de que, ao comprar os itens antecedentes, o consequente será comprado também.")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
else:
    print(f"\nNenhuma regra de associação foi encontrada com confiança mínima de {
          min_confidence}.")
    print("Sugestão: Tente reduzir o limite de confiança para gerar mais regras.")
