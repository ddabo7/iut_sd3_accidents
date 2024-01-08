import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# Charger le DataFrame depuis le fichier CSV
victime = pd.read_csv("step3/time_encoding.csv")

# Assure-toi que la colonne 'hrmn' est numérique
victime['hrmn'] = pd.to_numeric(victime['hrmn'], errors='coerce')

# Supprimer les lignes avec des valeurs manquantes dans 'hrmn'
victime = victime.dropna(subset=['hrmn'])

# Couper la colonne 'hrmn' en intervalles
hrmn_cut = pd.cut(victime['hrmn'], bins=24, labels=[str(i) for i in range(0, 24)])

# Remplacer la colonne 'hrmn' par les intervalles
victime['hrmn'] = hrmn_cut.values

# Extraire les coordonnées
X_lat = victime['lat']
X_long = victime['long']

# Créer un tableau avec les coordonnées
X_cluster = np.array(list(zip(X_lat, X_long)))

# Effectuer le clustering avec KMeans
clustering = KMeans(n_clusters=15, random_state=0)
clustering.fit(X_cluster)

# Ajouter les catégories dans le DataFrame
victime['geo'] = clustering.labels_

victime.to_csv("step4/gps_encoding.csv", index=False)
