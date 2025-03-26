import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Chargement des données normalisées
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

# Définition du modèle et des paramètres à tester
model = LinearRegression()
param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}

# GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Sauvegarde des meilleurs paramètres
best_params = grid_search.best_params_
import pickle
with open('models/best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)
