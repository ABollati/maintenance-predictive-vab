import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# data_piege = {
#     'id': [1, 2, 3, 3, 4, 5, 6], 
#     'km': [10000, 23000, 23000, "INC", -1200, 1500000, np.nan],
#     'etat': [0, 1, 1, 2, 2, 1, np.nan],
#     'panne': [1,0,0,0,1,0,1]
# }
# Note : 'INC' simule une erreur de saisie texte
# id 3 est un doublon
# 1 500 000 km est une aberration (supérieur à la limite de 1 000 000)

data_piege = {
    'id': [101, 102, 103, 104, 105],
    'km': [15000, 45000, 12000, 60000, 32000],
    'etat': [2, 1, 2, 0, 1],
    'panne': [0, 0, 0, 1, 0] # 1 = Oui, 0 = Non
}

df_piege = pd.DataFrame(data_piege)
#print(df_piege['panne'])

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
    #On enlève les valeurs NaN de "panne"
    df = df.dropna(subset=['panne'])
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

#2: NORMALISATION DES DONNEES

def normaliser_les_donnees(df):
    scaler = MinMaxScaler()
    df[['km','etat']] = scaler.fit_transform(df[['km','etat']])
    return df, scaler


#--------------------------------

#3: ENTRAINEMENT DU MODELE

def regression_logistique(df):
    df, scaler = normaliser_les_donnees(df)
    X = df[['km', 'etat']]
    y = df['panne']
    modele = LogisticRegression()
    modele.fit(X,y)
    return modele, scaler

#BLOC D'ORCHESTRATION DU SCRIPT

if __name__ == "__main__":
    
    print("--- DÉMARRAGE DU PIPELINE LOGISTIQUE ---")
    
    # 1. Chargement (On simule ou on charge un CSV)
    #df_raw = pd.read_csv("donnees_vab_brutes.csv") 
    
    #On crée une copie pour garder la donnée originale
    df = df_piege.copy()
    
    # 2. Nettoyage
    df_clean = nettoyer_donnees(df)
    #print(df_clean)
    
    # 3. Entraînement et Scaling
    mon_modele, mon_scaler = regression_logistique(df_clean)
    
   # Extraction des coefficients
    coefs = mon_modele.coef_[0]
    intercept = mon_modele.intercept_[0]

    print(f"Biais (Intercept) : {intercept:.2f}")
    print(f"Coefficient km : {coefs[0]:.2f}")
    print(f"Coefficient état : {coefs[1]:.2f}")
        
    # 4. Sauvegarde
    joblib.dump(mon_modele, "models/modele_final.pkl")
    joblib.dump(mon_scaler, "models/scaler_final.pkl")
    
    print("--- MISSION TERMINÉE : MODÈLE PARÉ À L'EMPLOI ---")




