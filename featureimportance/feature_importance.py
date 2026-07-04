from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "car_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "car_price_model.pkl"

# Load dataset
data = pd.read_csv(DATA_PATH)

# Encode categorical columns
le = LabelEncoder()
data["Car_Name"] = le.fit_transform(data["Car_Name"])
data["Fuel_Type"] = le.fit_transform(data["Fuel_Type"])
data["Seller_Type"] = le.fit_transform(data["Seller_Type"])
data["Transmission"] = le.fit_transform(data["Transmission"])

# Features
X = data.drop("Selling_Price", axis=1)

# Load trained model
model = joblib.load(MODEL_PATH)

# Print feature importance
importance = model.feature_importances_

for feature, score in zip(X.columns, importance):
    print(f"{feature}: {score:.4f}")