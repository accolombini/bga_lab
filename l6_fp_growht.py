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

    |||> Como alternativa vamos usar o FP-GROWHT  e gerar minimamnte os maiores e os menores suportes.

    |> Nota: A razão pela qual não conseguimos ver regras de associação explícitas na saída do FP-Growth é que o FP-Growth, por si só, não gera regras de associação – ele apenas identifica itemsets frequentes (conjuntos de itens que aparecem frequentemente juntos). O algoritmo FP-Growth constrói uma árvore de padrões frequentes (FP-Tree) para encontrar esses conjuntos, mas não transforma automaticamente esses itemsets em regras. Para obter regras de associação a partir dos itemsets frequentes gerados pelo FP-Growth, precisamos aplicar um passo adicional: geração de regras de associação.
    Como Gerar Regras de Associação com FP-Growth
    Após identificarmos os itemsets frequentes com o FP-Growth, podemos gerar regras de associação usando uma métrica, como confiança ou lift, para avaliar a força dessas regras. O pacote mlxtend, que estamos usando, oferece uma função chamada association_rules que podemos aplicar aos itemsets frequentes para obter regras de associação.
'''

# Importar as bibliotecas necessárias
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
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

# Discretizar as variáveis contínuas em categorias
# Cada variável contínua é dividida em 3 faixas: baixa, média e alta
# Esta discretização facilita a aplicação do FP-Growth em dados contínuos
for col in data.columns[:-1]:  # Exclui a coluna de classe, pois já está categorizada
    data[col] = pd.cut(data[col], bins=3, labels=[
                       f"{col}_low", f"{col}_med", f"{col}_high"])

# Transformar o dataset em uma lista de transações
# Cada linha do DataFrame é convertida em uma lista de itens discretizados
# Exemplo: ['fixed acidity_low', 'citric acid_med', ...]
transactions = data.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

# Codificar as transações para uso no FP-Growth
# TransactionEncoder converte a lista de itens em uma matriz binária onde:
# - Cada coluna representa um item
# - Cada linha representa uma transação (1 indica presença do item, 0 ausência)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Aplicar o FP-Growth para encontrar itemsets frequentes
# min_support define o suporte mínimo necessário para que um itemset seja considerado frequente
# Ajuste este parâmetro conforme necessário para encontrar mais ou menos padrões
min_support = 0.05
frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

# Ordenar os itemsets por suporte e exibir os 10 mais frequentes e os 10 menos frequentes
# Isso ajuda a identificar os padrões mais e menos comuns no dataset
top_frequent_itemsets = frequent_itemsets.sort_values(
    by="support", ascending=False).head(10)
bottom_frequent_itemsets = frequent_itemsets.sort_values(
    by="support", ascending=True).head(10)

# Reformatação dos itemsets para uma visualização mais clara
# Converte cada conjunto de itens em uma string de itens separados por vírgula
top_frequent_itemsets['itemsets'] = top_frequent_itemsets['itemsets'].apply(
    lambda x: ', '.join(list(x)))
bottom_frequent_itemsets['itemsets'] = bottom_frequent_itemsets['itemsets'].apply(
    lambda x: ', '.join(list(x)))

# Exibir os itemsets mais frequentes
print("Padrões Frequentes (Top 10 - Maior Suporte):")
print("Cada linha exibe um conjunto de características discretizadas com um suporte alto na base de dados.\n")
print(top_frequent_itemsets.to_string(
    index=False, header=["Suporte", "Conjunto de Itens"]))

# Exibir os itemsets menos frequentes
print("\nPadrões Frequentes (Bottom 10 - Menor Suporte):")
print("Cada linha exibe um conjunto de características discretizadas com um suporte baixo na base de dados.\n")
print(bottom_frequent_itemsets.to_string(
    index=False, header=["Suporte", "Conjunto de Itens"]))

'''
# Gerar regras de associação a partir dos itemsets frequentes
# min_confidence define a confiança mínima para considerar uma regra válida
min_confidence = 0.3  # Ajuste conforme necessário para obter mais ou menos regras
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Filtrar as colunas de interesse para exibir as regras de associação e criar uma cópia explícita
# A cópia evita warnings sobre modificações diretas em fatias do DataFrame
rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()

# Reformatação dos antecedentes e consequentes para uma visualização mais clara
# Converte cada conjunto de antecedentes/consequentes em uma string de itens separados por vírgula
rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))

# Exibir as regras de associação
print("\nRegras de Associação:")
print("Cada linha exibe uma regra que relaciona um conjunto de características (antecedentes) com uma consequência (consequentes).\n")
print(rules_display.to_string(index=False, header=["Antecedentes", "Consequentes", "Suporte", "Confiança", "Lift"]))
'''
