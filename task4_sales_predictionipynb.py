



# ==========================================
# ADVERTISING SALES PREDICTION (FIXED FULL CODE)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")

# ==========================================
# 1. LOAD DATA (FIXED WITH FALLBACK)
# ==========================================

try:
    df = pd.read_csv("Advertising.csv")  # local file
    print("Dataset loaded from CSV file successfully!")

except FileNotFoundError:
    print("CSV not found! Using synthetic advertising dataset...")

    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=200,
        n_features=3,
        noise=10,
        random_state=42
    )

    df = pd.DataFrame(X, columns=["TV", "Radio", "Newspaper"])
    df["Sales"] = y

print(df.head())
print(df.columns)

# ==========================================
# 2. CLEANING
# ==========================================

df.dropna(inplace=True)

# ==========================================
# 3. EDA (OPTIONAL VISUALIZATION)
# ==========================================

sns.pairplot(df)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# ==========================================
# 4. SPLIT DATA
# ==========================================

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 5. MODEL TRAINING
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# 6. PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 7. EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ==========================================
# 8. VISUALIZATION
# ==========================================

plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()