'''
# Importar bibliotecas necessáriasObjetivo: passos para criar um script utilizando o algoritmo Apriori.
    Para estes exemplo utilizaremos o conjunto de dados públiocos Online Retail Dataset (Online Retail) do UCI Machine Learning Repository. Nete exemplo, utilizaremos a base full.
'''

# Importar as bibliotecas necessárias
import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import requests

# Verificar se o arquivo "Online Retail.xlsx" já está disponível localmente; caso contrário, fazer o download
file_path = "Online Retail.xlsx"
if not os.path.exists(file_path):
    print("Baixando o arquivo Online Retail.xlsx...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    response = requests.get(url)
    # Salvar o conteúdo baixado no arquivo "Online Retail.xlsx"
    with open(file_path, 'wb') as f:
        f.write(response.content)
    print("Download concluído.")

# Carregar o dataset de vendas a partir do arquivo Excel
# Este dataset contém transações de vendas de uma loja online
data = pd.read_excel(file_path)

# Pré-processamento dos dados para preparar o conjunto de transações
# Remover linhas com valores ausentes nas colunas 'InvoiceNo' e 'Description'
data.dropna(subset=['InvoiceNo', 'Description'], inplace=True)

# Remover transações que são devoluções
# Identificadas pelo prefixo 'C' no número da nota fiscal (InvoiceNo)
data = data[~data['InvoiceNo'].str.contains('C', na=False)]

# Agrupar os dados por transação (InvoiceNo) para que cada transação seja uma lista de itens comprados juntos
# Cada transação será representada por uma lista de descrições de itens
basket = data.groupby(['InvoiceNo'])['Description'].apply(list)

# Converter os itens para strings para evitar problemas com o TransactionEncoder
basket = data.groupby(['InvoiceNo'])['Description'].apply(
    lambda x: list(x.astype(str)))

# Transformar os dados para o formato esperado pelo algoritmo Apriori
# O TransactionEncoder converte a lista de itens para uma matriz binária:
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
