import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Chargement des données
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

# Chargement des meilleurs paramètres
with open('models/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

# Entraînement de notre modèle
model = LinearRegression(**best_params)
model.fit(X_train, y_train)

# Sauvegarde de notre modèle
with open('models/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
