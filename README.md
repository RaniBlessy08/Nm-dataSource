import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (replace with your actual dataset path or URL)
df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')

# Drop unnamed and null-heavy columns
df = df.dropna(axis=1, how='all')
df = df.drop(['Date', 'Time'], axis=1)

# Replace -200 (missing value marker in UCI dataset) with NaN
df = df.replace(-200, np.nan)
df = df.dropna()

# Define features and target
X = df.drop('C6H6(GT)', axis=1)  # Predicting Benzene concentration
y = df['C6H6(GT)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

# Visualization
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Benzene Levels")
plt.ylabel("Predicted Benzene Levels")
plt.title("Actual vs Predicted Benzene Concentration")
plt.show()
