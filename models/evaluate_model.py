import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json

# Chargement des données
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Chargement de notre  modèle entraîné
with open('models/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prédictions
y_pred = model.predict(X_test)

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sauvegarde des métriques
metrics = {'mse': mse, 'r2': r2}
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f)

# Sauvegarde des prédictions
predictions = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
predictions.to_csv('data/processed/predictions.csv', index=False)
