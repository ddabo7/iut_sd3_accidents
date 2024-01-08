import pandas as pd

victime = pd.read_csv("step2/missing_values_deleted.csv")
victime_cleaned = victime.dropna()
# Exclure les colonnes non numériques avant le calcul de la corrélation
numeric_columns = victime_cleaned.select_dtypes(include=['number']).columns
correlation_matrix = victime_cleaned[numeric_columns].corr()


correlation_matrix.to_csv("step3/time_encoding.csv", index=False)