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

# Importar bibliotecas necessárias
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Definir o caminho para o arquivo baixado
file_path = "Online Retail.xlsx"

# Carregar o dataset
data = pd.read_excel(file_path)

# Pré-processamento de dados
data.dropna(subset=['InvoiceNo', 'Description'], inplace=True)
data = data[~data['InvoiceNo'].str.contains(
    'C', na=False)]  # Remover devoluções

# Agrupar os dados para que cada transação seja uma lista de itens
# Convertendo os itens para strings para evitar problemas com o TransactionEncoder
basket = data.groupby(['InvoiceNo'])['Description'].apply(
    lambda x: list(x.astype(str)))

# Transformar dados para o formato esperado
te = TransactionEncoder()
te_ary = te.fit(basket).transform(basket)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Aplicar Apriori
# Ajuste o suporte conforme necessário
frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
print("Itemsets Frequentes (Itens mais comprados juntos):\n")
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
