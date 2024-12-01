#1. Analisar e pré-processar os dados fornecidos. 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'C:\Users\Amanda\OneDrive\Área de Trabalho\FASE4_CAP3\CAP3\seeds_dataset.txt'


data = pd.read_csv(file_path, sep=r'\s+', header=None)

data.columns = [
    "Area",
    "Perimeter",
    "Compactness",
    "Kernel Length",
    "Kernel Width",
    "Asymmetry Coefficient",
    "Kernel Groove Length",
    "Class"
]

print("Estatísticas Descritivas:")
print(data.describe())

print("\nGerando histogramas...")
data.hist(bins=10, figsize=(12, 8))
plt.suptitle("Distribuição das Características")
plt.show()

print("\nGerando boxplots...")
plt.figure(figsize=(12, 8))
sns.boxplot(data=data)
plt.title("Boxplots das Características")
plt.xticks(rotation=45)
plt.show()

print("\nGerando gráficos de dispersão...")
sns.pairplot(data, hue="Class", diag_kind="kde")
plt.suptitle("Relações entre Características", y=1.02)
plt.show()

print("\nValores ausentes por coluna:")
print(data.isnull().sum())
data.fillna(data.mean(), inplace=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.iloc[:, :-1])
scaled_df = pd.DataFrame(scaled_data, columns=data.columns[:-1])
scaled_df['Class'] = data['Class']
print("\nDados padronizados:\n", scaled_df.head())

scaled_df.to_csv(r'C:\Users\Amanda\OneDrive\Área de Trabalho\FASE4_CAP3\CAP3\seeds_dataset.csv', index=False)
print("\nDados padronizados salvos em: seeds_scaled.csv")