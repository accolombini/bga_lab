import pandas as pd

# Definir URL para o dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00365/Faults.NNA"

# Carregar o dataset
data = pd.read_csv(url, sep='\s+', header=None)
print(data.head())
