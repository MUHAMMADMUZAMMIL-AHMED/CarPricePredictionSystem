from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ===========================================================
# Paths
# ===========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = BASE_DIR / "dataset" / "car_dataset.csv"

MODEL = BASE_DIR / "model" / "car_price_model.pkl"

MODEL.parent.mkdir(exist_ok=True)

# ===========================================================
# Load Dataset
# ===========================================================

data = pd.read_csv(DATASET)

print("Dataset Loaded Successfully!\n")

print(data.head())

# ===========================================================
# Encode Categorical Columns
# ===========================================================

car_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()
seller_encoder = LabelEncoder()
trans_encoder = LabelEncoder()

data["Car_Name"] = car_encoder.fit_transform(data["Car_Name"])

data["Fuel_Type"] = fuel_encoder.fit_transform(data["Fuel_Type"])

data["Seller_Type"] = seller_encoder.fit_transform(data["Seller_Type"])

data["Transmission"] = trans_encoder.fit_transform(data["Transmission"])

# ===========================================================
# Features
# ===========================================================

X = data.drop("Selling_Price", axis=1)

y = data["Selling_Price"]

feature_names = X.columns

# ===========================================================
# Split Dataset
# ===========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

# ===========================================================
# Train Model
# ===========================================================

# model = DecisionTreeRegressor(

#     random_state=42,

#     max_depth=8

# )
model = DecisionTreeRegressor(
    random_state=42,
    max_depth=5,        # shallower
    min_samples_leaf=5  # more regularization
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

# ===========================================================
# Evaluation
# ===========================================================

mae = mean_absolute_error(y_test, prediction)

mse = mean_squared_error(y_test, prediction)

r2 = r2_score(y_test, prediction)

print("\nModel Trained Successfully!\n")

print("========== RESULTS ==========\n")

print(f"MAE : {mae:.2f}")

print(f"MSE : {mse:.2f}")

print(f"R2 Score : {r2:.2f}")

# ===========================================================
# Save Model
# ===========================================================

joblib.dump(model, MODEL)

print("\nModel Saved Successfully!")

print(MODEL)

# ===========================================================
# GRAPH 1
# Mileage vs Selling Price
# ===========================================================

plt.figure(figsize=(10,6))

plt.scatter(

    data["Kms_Driven"],

    data["Selling_Price"]

)

plt.title("Mileage vs Selling Price")

plt.xlabel("Kilometers Driven")

plt.ylabel("Selling Price")

plt.grid(True)

plt.show()

# ===========================================================
# GRAPH 2
# Actual vs Predicted
# ===========================================================

plt.figure(figsize=(10,6))

plt.plot(

    y_test.values[:30],

    label="Actual",

    linewidth=2

)

plt.plot(

    prediction[:30],

    label="Predicted",

    linewidth=2

)

plt.title("Actual vs Predicted Price")

plt.xlabel("Cars")

plt.ylabel("Selling Price")

plt.legend()

plt.grid(True)

plt.show()

# ===========================================================
# GRAPH 3
# Residual Plot
# ===========================================================

residuals = y_test - prediction

plt.figure(figsize=(10,6))

plt.scatter(

    prediction,

    residuals

)

plt.axhline(

    y=0,

    color="red"

)

plt.title("Residual Plot")

plt.xlabel("Predicted Price")

plt.ylabel("Residual Error")

plt.grid(True)

plt.show()

# ===========================================================
# GRAPH 4
# Feature Importance
# ===========================================================

importance = model.feature_importances_

plt.figure(figsize=(10,6))

plt.barh(

    feature_names,

    importance

)

plt.title("Feature Importance (Decision Tree)")

plt.xlabel("Importance")

plt.ylabel("Features")

plt.grid(True)

plt.show()

print("\nProject Finished Successfully!")