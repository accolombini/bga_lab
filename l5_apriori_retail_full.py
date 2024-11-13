'''
# Importar bibliotecas necessáriasObjetivo: passos para criar um script utilizando o algoritmo Apriori.
    Para estes exemplo utilizaremos o conjunto de dados públiocos Online Retail Dataset (Online Retail) do UCI Machine Learning Repository. Nete exemplo, utilizaremos a base full.
'''

import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import requests

# Verificar se o arquivo já existe; caso contrário, fazer o download
file_path = "Online Retail.xlsx"
if not os.path.exists(file_path):
    print("Baixando o arquivo Online Retail.xlsx...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    print("Download concluído.")

# Carregar o dataset
data = pd.read_excel(file_path)

# Pré-processamento de dados
data.dropna(subset=['InvoiceNo', 'Description'], inplace=True)
data = data[~data['InvoiceNo'].str.contains('C', na=False)]  # Remover devoluções

# Agrupar os dados para que cada transação seja uma lista de itens
basket = data.groupby(['InvoiceNo'])['Description'].apply(list)

# Agrupar os dados para que cada transação seja uma lista de itens
# Convertendo os itens para strings para evitar problemas com o TransactionEncoder
basket = data.groupby(['InvoiceNo'])['Description'].apply(
    lambda x: list(x.astype(str)))

# Transformar dados para o formato esperado
te = TransactionEncoder()
te_ary = te.fit(basket).transform(basket)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Aplicar Apriori
frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)  # Ajuste o suporte conforme necessário
print("Itemsets Frequentes (Itens mais comprados juntos):\n")
print(frequent_itemsets)

# Gerar e exibir as regras de associação
min_confidence = 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Exibir as regras de associação, se existirem
if not rules.empty:
    print("\nRegras de Associação (Relações encontradas entre os itens):\n")
    print("As regras mostram a confiança de que, ao comprar os itens antecedentes, o consequente será comprado também.")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
else:
    print(f"\nNenhuma regra de associação foi encontrada com confiança mínima de {min_confidence}.")
    print("Sugestão: Tente reduzir o limite de confiança para gerar mais regras.")
