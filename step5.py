import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from sklearn.tree import DecisionTreeRegressor

# Charger le DataFrame
victime = pd.read_csv("step4/gps_encoding.csv")

# Sélectionner la variable cible
y = victime['grav']

# Sélectionner les features
features = ['Num_Acc', 'catu', 'grav', 'sexe', 'trajet', 'secu', 'an_nais', 'senc',
       'catv', 'occutc', 'obs', 'obsm', 'choc', 'manv', 'an', 'mois', 'jour',
       'hrmn', 'lum', 'agg', 'int', 'atm', 'col', 'com', 'lat', 'long', 'dep',
       'catr', 'circ', 'nbv', 'vosp', 'prof', 'plan', 'surf', 'infra', 'situ',
       'env1', 'geo']

# Encoder les variables catégorielles
X_train_data = pd.get_dummies(victime[features].astype(str))

X_train_data.to_csv("step5/one_hot_encoding.csv", index=False)

# Diviser l'ensemble de données en train et test
X_train, X_test, y_train, y_test = train_test_split(X_train_data, y, test_size=0.2, random_state=42)

# Exporter l'échantillon train dans un fichier train.csv
X_train.to_csv("step6/train.csv", index=False)

# Exporter l'échantillon test dans un fichier test.csv
X_test.to_csv("step6/test.csv", index=False)

print(y_train.dtypes)

# Modèle d'arbre de décision
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Métriques d'évaluation
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions, average='weighted')
dt_recall = recall_score(y_test, dt_predictions, average='weighted')
dt_f1 = f1_score(y_test, dt_predictions, average='weighted')

print("Decision Tree Metrics:")
print(f"Accuracy: {dt_accuracy}")
print(f"Precision: {dt_precision}")
print(f"Recall: {dt_recall}")
print(f"F1 Score: {dt_f1}")

# Modèle de KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

# Métriques d'évaluation
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, average='weighted')
knn_recall = recall_score(y_test, knn_predictions, average='weighted')
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')

print("\nKNN Metrics:")
print(f"Accuracy: {knn_accuracy}")
print(f"Precision: {knn_precision}")
print(f"Recall: {knn_recall}")
print(f"F1 Score: {knn_f1}")

# Modèle de Régression logistique
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Métriques d'évaluation
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions, average='weighted')
lr_recall = recall_score(y_test, lr_predictions, average='weighted')
lr_f1 = f1_score(y_test, lr_predictions, average='weighted')

print("\nLogistic Regression Metrics:")
print(f"Accuracy: {lr_accuracy}")
print(f"Precision: {lr_precision}")
print(f"Recall: {lr_recall}")
print(f"F1 Score: {lr_f1}")

# Exporter le meilleur modèle (celui avec la meilleure précision)
best_model = None
best_precision = 0

for model, precision, trained_model in zip(['Decision Tree Regressor', 'KNN', 'Logistic Regression'], 
                                           [dt_precision, knn_precision, lr_precision],
                                           [dt_model, knn_model, lr_model]):
    if precision > best_precision:
        best_precision = precision
        best_model = model
        best_trained_model = trained_model

# Exporter le meilleur modèle
joblib.dump(best_trained_model, 'best_model.pkl')

print(f"\nBest Model: {best_model} (Saved as best_model.pkl)")