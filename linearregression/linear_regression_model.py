# ============================
# CAR PRICE PREDICTION
# USING LINEAR REGRESSION
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pathlib import Path

# ============================
# LOAD DATASET
# ============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "car_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!\n")
print(df.head())

# ============================
# ENCODE CATEGORICAL COLUMNS
# ============================

label_encoder = LabelEncoder()

df["Car_Name"] = label_encoder.fit_transform(df["Car_Name"])
df["Fuel_Type"] = label_encoder.fit_transform(df["Fuel_Type"])
df["Seller_Type"] = label_encoder.fit_transform(df["Seller_Type"])
df["Transmission"] = label_encoder.fit_transform(df["Transmission"])

# ============================
# INPUT & OUTPUT
# ============================

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ============================
# TRAIN TEST SPLIT
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================
# TRAIN MODEL
# ============================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ============================
# PREDICTION
# ============================

y_pred = model.predict(X_test)

# ============================
# EVALUATION
# ============================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== RESULTS ==========")
print("MAE :", round(mae,2))
print("MSE :", round(mse,2))
print("R2 Score :", round(r2,2))

# ============================
# GRAPH 1
# Mileage vs Selling Price
# ============================

plt.figure(figsize=(8,5))

plt.scatter(df["Kms_Driven"], df["Selling_Price"])

plt.title("Mileage vs Selling Price")
plt.xlabel("Kilometers Driven")
plt.ylabel("Selling Price")

plt.grid(True)

plt.show()

# ============================
# GRAPH 2
# Actual vs Predicted
# ============================

plt.figure(figsize=(8,5))

plt.plot(y_test.values[:30], label="Actual")
plt.plot(y_pred[:30], label="Predicted")

plt.title("Actual vs Predicted Price")
plt.xlabel("Cars")
plt.ylabel("Selling Price")

plt.legend()

plt.show()

# ============================
# GRAPH 3
# Residual Plot
# ============================

residuals = y_test - y_pred

plt.figure(figsize=(8,5))

plt.scatter(y_pred, residuals)

plt.axhline(y=0, color='red')

plt.title("Residual Plot")
plt.xlabel("Predicted Price")
plt.ylabel("Residual Error")

plt.grid(True)

plt.show()

# ============================
# GRAPH 4
# Feature Importance
# ============================

coefficients = pd.Series(model.coef_, index=X.columns)

coefficients.sort_values().plot(
    kind="barh",
    figsize=(10,6)
)

plt.title("Feature Coefficients")

plt.show()

print("\nProject Finished Successfully!")