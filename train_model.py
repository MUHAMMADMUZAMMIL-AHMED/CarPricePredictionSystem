from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ======================================
# Paths
# ======================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "car_dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "random_forest_model.pkl"

MODEL_PATH.parent.mkdir(exist_ok=True)

# ======================================
# Load Dataset
# ======================================

data = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!\n")
print(data.head())
print(data.info())
print(data.isnull().sum())

# ======================================
# Encode Categorical Columns
# ======================================

le = LabelEncoder()

data["Car_Name"] = le.fit_transform(data["Car_Name"])
data["Fuel_Type"] = le.fit_transform(data["Fuel_Type"])
data["Seller_Type"] = le.fit_transform(data["Seller_Type"])
data["Transmission"] = le.fit_transform(data["Transmission"])

# ======================================
# Features & Target
# ======================================

X = data.drop(columns=["Selling_Price"])
y = data["Selling_Price"]

# ======================================
# Train/Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ======================================
# Train Model
# ======================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

# ======================================
# Evaluation
# ======================================

mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print("\n========== RESULTS ==========")
print(f"MAE      : {mae:.2f}")
print(f"MSE      : {mse:.2f}")
print(f"R2 Score : {r2:.2f}")

# ======================================
# Save Model
# ======================================

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to:\n{MODEL_PATH}")
print("\nProject Finished Successfully!")