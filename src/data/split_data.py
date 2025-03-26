import pandas as pd
from sklearn.model_selection import train_test_split

# Chargement du dataset
data = pd.read_csv('data/raw/raw.csv')

# Séparation de la variable cible et des features
X = data.drop(columns=['silica_concentrate'])
y = data['silica_concentrate']

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarde des datasets traités
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)
