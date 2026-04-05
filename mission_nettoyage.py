import pandas as pd
import numpy as np

data = {
    'id': range(1, 11),
    'km': [10000, 12000, np.nan, 5000000, 15000, np.nan, 22000, -500, 18000, 25000],
    'etat': [2, 2, 1, 0, 2, np.nan, 1, 1, 2, 0]
}

df_sale = pd.DataFrame(data)
#print("--- Données brutes du terrain ---")
#print(df_sale)

#--------------------------------

# Nettoyage des données

# Remplacement des km manquants par la médiane des km
df_sale['km'] = df_sale['km'].fillna(df_sale['km'].median())

# On supprime les lignes avec des km supérieurs à 1000000
df_sale = df_sale[df_sale['km'] <= 1000000]

# On supprime les lignes avec des km négatifs
df_sale = df_sale[df_sale['km'] >= 0]

# On remplace les états manquants par 2 (bon par défaut)
df_sale['etat'] = df_sale['etat'].fillna(2)

#print("\n---Données nettoyées---")
#print(df_sale)

#--------------------------------

# Visualisation du nombre de VAB par km

import matplotlib.pyplot as plt

# # Création du graphique
# plt.figure(figsize=(10, 6))
# plt.hist(df_sale['km'], bins=5, color='skyblue', edgecolor='black')

# # Habillage (très important pour le commandement)
# plt.title("Répartition du kilométrage de la flotte VAB")
# plt.xlabel("Kilomètres")
# plt.ylabel("Nombre de véhicules")
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Affichage
# plt.show()

#--------------------------------

# Visualisation multidimensionnelle du nombre de VAB par km et par état

# Nouveau graphique : Nuage de points
plt.figure(figsize=(10, 6))
plt.scatter(df_sale['km'], df_sale['etat'], color='red', s=100, marker='x')

# Habillage
plt.title("Corrélation KM vs État du moteur")
plt.xlabel("Kilomètres")
plt.ylabel("État (0=Critique, 2=Bon)")
plt.yticks([0, 1, 2]) # Pour n'afficher que les notes entières
plt.grid(True, linestyle=':', alpha=0.6)

plt.show()
