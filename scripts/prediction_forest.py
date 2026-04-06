import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Chargement du "cerveau" sauvegardé
modele_rf = joblib.load('models/modele_forest.pkl')


#BLOC D'ORCHESTRATION DU SCRIPT

if __name__ == "__main__":
    
    print("--- SYSTÈME DE PRÉDICTION DE PANNE VAB PAR FORÊTS ALEATOIRES ---")
    
    
    # # Simulation d'une saisie terrain
    km = float(input("Entrez le kilométrage actuel : "))
    etat = int(input("Entrez l'état (0:Mauvais, 1:Moyen, 2:Bon) : "))
    
    nouveau_vab = pd.dataframe([[km,etat]], columns = ['km','etat'])
    
    proba_rf = modele_rf.predict_proba(nouveau_vab)[0][1]
    print(f"Probabilité de panne (Forêt) : {proba_rf:.2%}")
