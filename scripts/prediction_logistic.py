import joblib
import pandas as pd

FEATURES = ['km', 'condition', 'vehicle_age', 'num_revisions', 'engine_temperature']

model     = joblib.load('models/model_logistic.pkl')
scaler    = joblib.load('models/scaler_logistic.pkl')
threshold = joblib.load('models/threshold_logistic.pkl')


def predict_breakdown(km, condition, vehicle_age, num_revisions, engine_temperature):
    data = pd.DataFrame([[km, condition, vehicle_age, num_revisions, engine_temperature]],
                        columns=FEATURES)
    data_scaled = pd.DataFrame(scaler.transform(data), columns=FEATURES)
    proba = model.predict_proba(data_scaled)[0][1]
    prediction = int(proba >= threshold)
    return prediction, proba


if __name__ == "__main__":
    print("--- VAB BREAKDOWN PREDICTION — LOGISTIC REGRESSION ---")
    print(f"Decision threshold: {threshold:.3f}")

    km = float(input("Enter current mileage (km): "))
    condition = int(input("Enter engine condition (0=Critical, 1=Fair, 2=Good): "))
    vehicle_age = int(input("Enter vehicle age (years): "))
    num_revisions = int(input("Enter number of past revisions: "))
    engine_temperature = int(input("Enter engine temperature (°C): "))

    verdict, score = predict_breakdown(km, condition, vehicle_age, num_revisions, engine_temperature)

    if verdict == 1:
        print(f"ALERT: High breakdown risk ({score:.2%}). Maintenance required.")
    else:
        print(f"OK: Vehicle operational. Breakdown probability: {score:.2%}")
