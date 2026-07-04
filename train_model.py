from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "car_dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "car_price_model.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)

data = pd.read_csv(DATA_PATH)

print(data.head())
print(data.info())
print(data.isnull().sum())

le = LabelEncoder()
data["Car_Name"] = le.fit_transform(data["Car_Name"])
data["Fuel_Type"] = le.fit_transform(data["Fuel_Type"])
data["Seller_Type"] = le.fit_transform(data["Seller_Type"])
data["Transmission"] = le.fit_transform(data["Transmission"])

X = data.drop(columns=["Selling_Price"])
y = data["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
prediction = model.predict(X_test)

print("R2 Score:", r2_score(y_test, prediction))
print("MAE:", mean_absolute_error(y_test, prediction))
joblib.dump(model, MODEL_PATH)
print("Model saved to:", MODEL_PATH)