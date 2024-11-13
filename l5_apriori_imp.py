'''
    Objetivo: passos para criar um script utilizando o algoritmo Apriori.
    Para estes exemplo utilizaremos o conjunto de dados públiocos Online Retail Dataset (Online Retail) do UCI Machine Learning Repository. Para efeito de demonstração (tempo de execução, vamos usar uma versão reduzida, chamada Dataset de Transações de Exemplo)

    Nota: Suporte de 0.75 para itens individuais significa que esses itens estão em 75% das transações.
        Suporte de 0.5 para conjuntos de itens significa que esses pares de itens aparecem juntos em 50% das transações.
        As regras de associação mostram a probabilidade de compra conjunta e a força dessa relação:
        Confiança mostra a probabilidade de o consequente ser comprado quando o antecedente é comprado.
        Lift indica a força da relação; neste caso, a relação é relativamente fraca, já que o lift está próximo de 1.
'''

# Importar as bibliotecas necessárias
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Dados de exemplo representando compras de itens (transações)
dataset = [
    ['Leite', 'Ovos'],
    ['Leite', 'Biscoitos'],
    ['Leite', 'Ovos', 'Biscoitos'],
    ['Ovos', 'Biscoitos']
]

# Transformação dos dados para o formato adequado
# TransactionEncoder transforma a lista de transações em uma matriz binária,
# onde cada coluna representa um item e cada linha uma transação (1 indica presença, 0 ausência)
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)  # Fit e transformação dos dados
# Conversão para DataFrame do Pandas
df = pd.DataFrame(te_ary, columns=te.columns_)

# Aplicar o algoritmo Apriori para identificar os itemsets frequentes
# min_support define a frequência mínima que um itemset deve ter para ser considerado frequente
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Exibir os itemsets frequentes encontrados pelo Apriori
print("Itemsets Frequentes (Itens mais comprados juntos):\n")
print("Suporte indica a proporção de transações que contêm o itemset.")
print("Quanto maior o suporte, mais frequente é o itemset.")
print(frequent_itemsets)

# Gerar regras de associação a partir dos itemsets frequentes
# As regras serão baseadas na confiança mínima definida em min_confidence
# A confiança mede a probabilidade de o consequente ser comprado quando o antecedente é comprado
min_confidence = 0.5  # Confiança mínima para considerar uma regra
rules = association_rules(
    frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Exibir as regras de associação, se existirem
if not rules.empty:
    print("\nRegras de Associação (Relações encontradas entre os itens):\n")
    print("Cada regra mostra a confiança de que, ao comprar os itens antecedentes, o consequente será comprado também.")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
else:
    print(f"\nNenhuma regra de associação foi encontrada com confiança mínima de {
          min_confidence}.")
    print("Sugestão: Tente reduzir o limite de confiança para gerar mais regras.")
