import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve

from data_utils import FEATURES, TARGET, load_data, clean_data


def train_random_forest(df):
    X = df[FEATURES]
    y = df[TARGET]
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, X, y


def display_feature_importance(model):
    print("Feature importances:")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.2%}")


def compute_optimal_threshold(X, y):
    model_cv = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    y_proba = cross_val_predict(model_cv, X, y, cv=5, method='predict_proba')[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    return float(thresholds[np.argmax(f1)])


if __name__ == "__main__":

    print("--- RANDOM FOREST PIPELINE STARTED ---")

    df_raw = load_data()
    df_clean = clean_data(df_raw.copy())

    model, X, y = train_random_forest(df_clean)
    display_feature_importance(model)

    threshold = compute_optimal_threshold(X, y)
    print(f"Optimal threshold (F1-maximising, 5-fold CV): {threshold:.3f}")

    joblib.dump(model, "models/model_forest.pkl")
    joblib.dump(threshold, "models/threshold_forest.pkl")

    print("--- PIPELINE COMPLETE: MODEL AND THRESHOLD SAVED ---")
