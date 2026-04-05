# Maintenance Prédictive - VAB

Ce projet vise à prédire la probabilité et l'occurrence de panne de Véhicules de l'Avant Blindés (VAB) en fonction de leur kilométrage et de l'état du moteur, à l'aide de modèles de type régression logistique et forêt aléatoire.
Il s'agit d'un problème de classification par apprentissage supervisé.
Les données sont fictives et servent de démonstration technique.

## Modélisation

J'ai comparé deux modèles d'apprentissage automatique :
1. **Régression Logistique** : Pour son explicabilité (analyse des coefficients)
2. **Forêt aléatoire** : Pour sa robustesse face aux données atypiques

## Données

Le projet s'appuie sur un jeu de données de maintenance (format CSV). 
1. **Fichier source** : `data/donnees_vab_brutes.csv`
2. **Variables clés** : 
   1. `km` : Kilométrage au compteur.
   2. `etat` : État moteur (0: Critique, 1: Moyen, 2: Bon).
   3. `panne` : Variable cible (0: RAS, 1: Panne détectée).

> **Note technique** : Les scripts disposent d'une sécurité (fallback). Si le fichier CSV est absent, un jeu de données de test est automatiquement généré pour permettre l'exécution du code sans erreur. Ce jeu de données est celui qui a été utilisé pour généré les modèles présents dans le répertoire `models/`

## Installation

1. **Cloner le répertoire** :
   ```bash
   git clone https://github.com/ABollati/maintenance-predictive-vab.git
   ```
   
2. **Entrer dans le projet** :
   ```bash
   cd maintenance-predictive-vab
   ```

3. **Installer les dépendances** :
    Assurez-vous d'avoir Python installé, puis lancez :
    ```bash
    pip install -r requirements.txt
    ```

## Travaux de Recherche
Le dossier `research/` contient les scripts de prototypage (`mission_vab`, `mission_nettoyage`) ayant servi à valider les briques logiques avant leur intégration dans les pipelines finaux.

## Utilisation

Le projet est divisé en deux phases : l'entraînement des modèles et la prédiction en conditions réelles.

1. **Entraînement (Pipelines)** :
    Pour ré-entraîner les modèles avec de nouvelles données historiques :
    ```bash
    python scripts/pipeline_logistique.py
    python scripts/pipeline_forest.py
    ```
    
2. **Prédiction (Terrain)** :
   Pour tester la probabilité de panne d'un véhicule spécifique, lancez l'un des deux scripts "prediction_logistique.py" ou "prediction_forest.py":
   ```bash
   python scripts/prediction_logistique.py
   ```
   Le système vous demandera alors de saisir le kilométrage et l'état du moteur (0: Critique, 1: Moyen, 2: Bon).
