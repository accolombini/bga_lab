'''
    Importar bibliotecas necessáriasObjetivo: passos para criar um script utilizando o algoritmo Apriori.
    Para estes exemplo utilizaremos o conjunto de dados públiocos Online Retail Dataset (Online Retail) do UCI Machine Learning Repository. Nete exemplo, utilizaremos a base full.

    Regras de Associação
    As regras de associação fornecem insights sobre a probabilidade de que, ao comprar determinados itens (antecedents), outros itens (consequents) também sejam comprados.

    Cada regra de associação apresenta:

    Support: Frequência da combinação de antecedents e consequents na base de dados.
    Confidence (Confiança): A probabilidade de que os itens em consequents sejam comprados quando os itens em antecedents são comprados.
    Lift: Mede a força da associação entre antecedents e consequents. Valores acima de 1 indicam uma relação positiva.

        |||> Uma simples analise:

            (60 TEATIME FAIRY CAKE CASES) -> (PACK OF 72 RETROSPOT CAKE CASES)

            Confiança: 0.547 – Indica que, em 54,7% das compras que incluem "60 TEATIME FAIRY CAKE CASES", "PACK OF 72 RETROSPOT CAKE CASES" também é comprado.
            Lift: 8.54 – Uma associação forte, indicando que esses produtos tendem a ser comprados juntos bem mais do que o esperado aleatoriamente.
            (ALARM CLOCK BAKELIKE PINK) -> (ALARM CLOCK BAKELIKE GREEN)

            Confiança: 0.533 – Em 53,3% das transações que incluem o relógio "ALARM CLOCK BAKELIKE PINK", também aparece o "ALARM CLOCK BAKELIKE GREEN".
            Lift: 11.22 – Esse lift alto sugere uma forte associação, provavelmente devido à popularidade dos itens de cores diferentes sendo comprados juntos.
'''

# Importar as bibliotecas necessárias
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Definir o caminho para o arquivo de dados
# Aqui, assume-se que o arquivo "Online Retail.xlsx" está no mesmo diretório do script
file_path = "Online Retail.xlsx"

# Carregar o dataset de vendas
# Este arquivo contém transações de compras de uma loja online
data = pd.read_excel(file_path)

# Pré-processamento de dados para limpeza e preparação
# Remover linhas com valores ausentes nas colunas 'InvoiceNo' e 'Description'
data.dropna(subset=['InvoiceNo', 'Description'], inplace=True)

# Remover transações que são devoluções
# Identificadas pelo prefixo 'C' no número da nota fiscal (InvoiceNo)
data = data[~data['InvoiceNo'].str.contains('C', na=False)]

# Agrupar os dados por cada transação (InvoiceNo) e listar os itens (Description) comprados juntos
# Convertemos os itens para strings para evitar problemas de codificação com o TransactionEncoder
basket = data.groupby(['InvoiceNo'])['Description'].apply(
    lambda x: list(x.astype(str)))

# Transformar os dados para o formato esperado pelo algoritmo Apriori
# TransactionEncoder converte a lista de itens para uma matriz binária, onde:
# - Cada coluna representa um item
# - Cada linha representa uma transação (1 indica presença do item, 0 ausência)
te = TransactionEncoder()
te_ary = te.fit(basket).transform(basket)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Aplicar o algoritmo Apriori para identificar os itemsets frequentes
# min_support define o suporte mínimo necessário para que um itemset seja considerado frequente
# Quanto maior o suporte, mais comum é o itemset
frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
print("Itemsets Frequentes (Itens mais comprados juntos):\n")
print(frequent_itemsets)

# Gerar regras de associação a partir dos itemsets frequentes
# As regras serão baseadas na confiança mínima definida em min_confidence
# Confiança é a probabilidade de o consequente ser comprado, dado que o antecedente foi comprado
min_confidence = 0.5  # Confiança mínima para considerar uma regra válida
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
