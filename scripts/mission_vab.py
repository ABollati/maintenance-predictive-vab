import pandas as pd
import numpy as np

# Création de données fictives : 5 véhicules
data = {
    'id_vehicule': [101, 102, 103, 104, 105],
    'km_compteur': [15000, 45000, 12000, 60000, 32000],
    'derniere_revision_jours': [30, 180, 10, 400, 90],
    'etat_moteur': ['Bon', 'Moyen', 'Bon', 'Critique', 'Moyen'],
    'en_mission': [1, 0, 1, 0, 1] # 1 = Oui, 0 = Non
}

df = pd.DataFrame(data)
#print("--- État initial de la flotte ---")
#print(df)

# Isoler les véhicules dont l'état est 'Critique'
vehicules_en_danger = df[df['etat_moteur'] == 'Critique']

#print("\n--- ALERTE : MATÉRIEL EN ÉTAT CRITIQUE ---")
#print(vehicules_en_danger)

# Dictionnaire de traduction (Mapping)
traduction = {'Critique': 0, 'Moyen': 1, 'Bon': 2}

# Application de la transformation
df['etat_num'] = df['etat_moteur'].map(traduction)

#print("\n--- Données numérisées pour l'IA ---")
#print(df[['id_vehicule', 'etat_moteur', 'etat_num']])

#print("\n--- État initial de la flotte numérisée pour l'IA ---")
#print(df[['id_vehicule', 'km_compteur', 'derniere_revision_jours', 'etat_num','en_mission']])

#--------------------------------
#Adimensionnement des données
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 1. Initialisation du "Scaler"
scaler = MinMaxScaler()

# 2. Apprentissage et Transformation
# On transforme les colonnes pour qu'elles soient entre 0 et 1
df[['km_compteur', 'etat_num']] = scaler.fit_transform(df[['km_compteur', 'etat_num']])

#print("--- Données Normalisées (Prêtes pour l'IA) ---")
#print(df)

#--------------------------------
#Régression logistique

from sklearn.linear_model import LogisticRegression

# Données d'entraînement (X = caractéristiques, y = résultat)
# On imagine des données historiques : km, etat_num
X = df[['km_compteur', 'etat_num']]
# Cible : Est-ce qu'il est tombé en panne ? (données fictives pour l'exemple)
y = [0, 0, 0, 1, 0] 

# Création et entraînement du modèle
modele = LogisticRegression()
modele.fit(X, y)

# Test : Si un nouveau véhicule arrive avec 55 000 km et un état 0 (Critique)
nouveau_vab = pd.DataFrame([[400000, 0]], columns=['km_compteur', 'etat_num'])
nouveau_vab_norm = pd.DataFrame(scaler.transform(nouveau_vab), columns=nouveau_vab.columns)
prediction = modele.predict(nouveau_vab_norm)
probabilite = modele.predict_proba(nouveau_vab_norm)

#print(f"\nPrédiction de panne pour le nouveau véhicule : {'PANNE' if prediction[0] == 1 else 'RAS'}")
print(f"\nProbabilité de panne pour le nouveau véhicule :{probabilite[0][1]}")

# Extraction des coefficients
coefs = modele.coef_[0]
intercept = modele.intercept_[0]

print(f"Biais (Intercept) : {intercept:.2f}")
print(f"Coefficient km_compteur : {coefs[0]:.2f}")
print(f"Coefficient etat_num : {coefs[1]:.2f}")
