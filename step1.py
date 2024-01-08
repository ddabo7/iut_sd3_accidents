import pandas as pd

# Charger les fichiers CSV
carac = pd.read_csv("data/carac.csv",sep=';')
lieux = pd.read_csv("data/lieux.csv", sep=';', dtype={'catr': 'float'})
veh = pd.read_csv("data/veh.csv",sep=';')
vict = pd.read_csv("data/vict.csv",sep=';')

# Fusionner les fichiers en fonction des clés appropriées
# Assure-toi d'ajuster les noms de colonnes selon les clés de fusion
victime = vict.merge(veh,on=['Num_Acc','num_veh'])
accident = carac.merge(lieux,on = 'Num_Acc')
victime = victime.merge(accident,on='Num_Acc')

# Afficher le résultat
print(victime)

# Sauvegarder le fichier fusionné
victime.to_csv("step1/merged_data.csv/victime.csv", index=False) 