import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve

from data_utils import FEATURES, TARGET, load_data, clean_data


def scale_features(df):
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    return df, scaler


def display_coefficients(model):
    print("Logistic Regression coefficients:")
    coefs = model.coef_[0]
    for name, coef in sorted(zip(FEATURES, coefs), key=lambda x: -abs(x[1])):
        print(f"  {name}: {coef:.2f}")


def train_logistic_regression(df):
    df, scaler = scale_features(df)
    X = df[FEATURES]
    y = df[TARGET]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, scaler, X, y


def compute_optimal_threshold(X, y):
    pipeline_cv = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    y_proba = cross_val_predict(pipeline_cv, X, y, cv=5, method='predict_proba')[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    return float(thresholds[np.argmax(f1)])


if __name__ == "__main__":

    print("--- LOGISTIC REGRESSION PIPELINE STARTED ---")

    df_raw = load_data()
    df_clean = clean_data(df_raw.copy())

    model, scaler, X, y = train_logistic_regression(df_clean)
    display_coefficients(model)

    threshold = compute_optimal_threshold(X, y)
    print(f"Optimal threshold (F1-maximising, 5-fold CV): {threshold:.3f}")

    joblib.dump(model, "models/model_logistic.pkl")
    joblib.dump(scaler, "models/scaler_logistic.pkl")
    joblib.dump(threshold, "models/threshold_logistic.pkl")

    print("--- PIPELINE COMPLETE: MODEL, SCALER AND THRESHOLD SAVED ---")
