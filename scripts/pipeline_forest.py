import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib


#1: NETTOYAGE DES DONNEES

#Suppression des doublons

def supprimer_doublons(df):
    return df.drop_duplicates(subset=['id'], keep='first')

#Conversion des types en nombre

def conversion_en_nombre(df):
    df.loc[:,'km'] = pd.to_numeric(df['km'], errors='coerce')
    df.loc[:,'etat'] = pd.to_numeric(df['etat'], errors='coerce')
    return df

#Traitement des valeurs manquantes (NaN)

def traitement_des_valeurs_NaN(df):
    # Sécurité : Si km est tout vide, on met 0 ou on prévient
    if df['km'].isnull().all():
        print("ATTENTION : Colonne km vide, remplacement par 0")
        df['km'] = df['km'].fillna(0)
    else:
        median_km = df['km'].median()
        df.loc[:,'km'] = df['km'].fillna(median_km)
    
    # État : 2 par défaut
    df['etat'] = df['etat'].fillna(2)
    return df

#Filtrage des aberrations

def filtrage_des_valeurs_aberrantes(df):
    return df[(df['km'] <= 1000000) & (df['km'] >= 0)]

#Fonction complète

def nettoyer_donnees(df):
    df_sans_doublons = supprimer_doublons(df)
    df_numerique = conversion_en_nombre(df_sans_doublons)
    df_NaN_to_median = traitement_des_valeurs_NaN(df_numerique)
    df_nettoye = filtrage_des_valeurs_aberrantes(df_NaN_to_median)
    return df_nettoye


#--------------------------------

#2: ENTRAINEMENT DU MODELE

def entrainer_foret(df):
    # X = Caractéristiques, y = Cible
    X = df[['km', 'etat']]
    y = df['panne']
    
    # Initialisation de la forêt
    # n_estimators=100 : 100 arbres vont voter
    # max_depth=3 : On limite la croissance pour éviter l'overfitting (vu votre peu de données)
    modele_rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    #print("Valeurs manquantes dans y :", y.isna().sum())
    modele_rf.fit(X, y)
    return modele_rf

# Extraction de l'importance des paramètres
def afficher_importance(modele):
    importances = modele.feature_importances_
    print(f"Importance km: {importances[0]:.2%}")
    print(f"Importance état: {importances[1]:.2%}")

#BLOC D'ORCHESTRATION DU SCRIPT

if __name__ == "__main__":
    
    print("--- DÉMARRAGE DU PIPELINE FORÊTS ALEATOIRES ---")
    
    # 1. Chargement (On simule ou on charge un CSV)
    def charger_donnees():
        try:
            df_raw = pd.read_csv('data/donnees_brutes_vab.csv')
            print("Chargement depuis le CSV réussi.")
        except FileNotFoundError:
            print("CSV introuvable. Génération du jeu de données de secours...")
            data_exception = {
                'id': [101, 102, 103, 104, 105],
                'km': [15000, 45000, 12000, 60000, 32000],
                'etat': [2, 1, 2, 0, 1],
                'panne': [0, 0, 0, 1, 0]
            }
            df_raw = pd.DataFrame(data_exception)
        return df_raw

    df_raw = charger_donnees()
    
    #Copie des données pour garder la dataframe originale
    df = df_raw.copy()
    
    # 2. Nettoyage
    df_clean = nettoyer_donnees(df)
    #print(df_clean)
    
    # 3. Entraînement
    modele_rf = entrainer_foret(df_clean)
    
    # 4. Importance des parametres
    afficher_importance(modele_rf)
    
    # On sauvegarde le modèle entraîné
    joblib.dump(modele_rf, 'models/modele_forest.pkl')
    print("Modèle Forêt Aléatoire sauvegardé avec succès.")
    
    print("--- MISSION TERMINÉE : MODÈLE PARÉ À L'EMPLOI ---")




