import joblib
import pandas as pd
import numpy as np

# 1. Chargement des "cerveaux" sauvegardés
modele = joblib.load('models/modele_final.pkl')
scaler = joblib.load('models/scaler_final.pkl')

def predire_panne(km_brut, etat_brut):
    # On met les données dans le bon format pour le scaler
    donnees = pd.DataFrame([[km_brut, etat_brut]],columns=['km', 'etat'])
    
    # ÉTAPE CRUCIALE : On normalise avec l'étalon du passé
    donnees_scalees = pd.DataFrame(scaler.transform(donnees),columns=donnees.columns)
    
    # On demande l'avis au modèle
    prediction = modele.predict(donnees_scalees)
    probabilite = modele.predict_proba(donnees_scalees)
    
    return prediction[0], probabilite[0][1]

if __name__ == "__main__":
    print("--- SYSTÈME DE PRÉDICTION DE PANNE VAB PAR REGRESSION LOGISTIQUE ---")
    
    # Simulation d'une saisie terrain
    km = float(input("Entrez le kilométrage actuel : "))
    etat = int(input("Entrez l'état (0:Mauvais, 1:Moyen, 2:Bon) : "))
    
    verdict, score = predire_panne(km, etat)
    
    if verdict == 1:
        print(f"ALERTE : Risque de panne élevé ({score:.2%}) ! Maintenance requise.")
    else:
        print(f"RAS : Véhicule opérationnel. Confiance : {(1-score):.2%}")
