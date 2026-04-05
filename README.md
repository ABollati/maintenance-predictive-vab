# Maintenance Prédictive - VAB

Ce projet vise à prédire la probabilité et l'occurrence de panne de Véhicules de l'Avant Blindés (VAB) en fonction de leur kilométrage et de l'état du moteur, à l'aide de modèles de type régression logistique et forêt aléatoire.
Il s'agit d'un problème de classification par apprentissage supervisé.
Les données sont fictives et servent de démonstration technique.

## Approche

J'ai comparé deux modèles d'apprentissage automatique :
1. **Régression Logistique** : Pour son explicabilité (analyse des coefficients)
3. **Forêt aléatoire** : Pour sa robustesse face aux données atypiques

## Installation

1. **Cloner le répertoire** :
   ```bash
   git clone [https://github.com/ABollati/maintenance-predictive-vab.git](https://github.com/ABollati/maintenance-predictive-vab.git)
   cd maintenance-predictive-vab
   ```
 2. **Installer les dépendances** :
    Assurez-vous d'avoir Python installé, puis lancez :
    ```bash
    pip install -r requirements.txt
    ```
## Utilisation

Le projet est divisé en deux phases : l'entraînement des modèles et la prédiction en conditions réelles.

1. **Entraînement (Pipelines)** :
    Pour ré-entraîner les modèles avec de nouvelles données historiques :
    ```bash
    python scripts/pipeline_logistique.py
    python scripts/pipeline_forest.py
    ```
    
2. **Prédiction (Terrain)** :
   Pour tester la probabilité de panne d'un véhicule spécifique, lancez l'un des scripts, par exemple :
   ```bash
   python scripts/prediction_logistique.py
   ```
   Le système vous demandera alors de saisir le kilométrage et l'état du moteur (0: Critique, 1: Moyen, 2: Bon).
