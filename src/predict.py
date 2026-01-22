import pandas as pd
import pickle
from features import prepare_features

df = pd.read_csv("data/student_data.csv")

with open("model/risk_model.pkl", "rb") as f:
    model = pickle.load(f)

X, _ = prepare_features(df)

preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]  # probability of class 1 (at risk)

out = df.copy()
out["predicted_risk_status"] = preds
out["risk_probability"] = probs

out.to_csv("data/predictions.csv", index=False)
print("Saved predictions to data/predictions.csv")
print(model)
