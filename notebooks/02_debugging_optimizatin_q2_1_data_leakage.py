"""
Q2.1: Data Leakage Bug
This model predicts all delivery times as 28 minutes. Debug and fix it.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv('/mnt/user-data/uploads/uber-eats-deliveries.csv')

# Clean target variable
df['Time_taken(min)'] = df['Time_taken(min)'].str.replace('(min) ', '').astype(float)

# Feature engineering
df['delivery_distance'] = np.sqrt(
    (df['Delivery_location_latitude'] - df['Restaurant_latitude'])**2 + 
    (df['Delivery_location_longitude'] - df['Restaurant_longitude'])**2
) * 111  # rough km conversion

df['hour'] = pd.to_datetime(df['Time_Orderd']).dt.hour
df['is_peak_hour'] = ((df['hour'] >= 12) & (df['hour'] <= 14)) | ((df['hour'] >= 19) & (df['hour'] <= 21))

# Encode categorical variables
df['Weather_encoded'] = df['Weatherconditions'].str.replace('conditions ', '').astype('category').cat.codes
df['Traffic_encoded'] = df['Road_traffic_density'].astype('category').cat.codes
df['Vehicle_encoded'] = df['Type_of_vehicle'].astype('category').cat.codes

# Select features
feature_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 
                'delivery_distance', 'hour', 'is_peak_hour',
                'Weather_encoded', 'Traffic_encoded', 'Vehicle_encoded',
                'Vehicle_condition', 'multiple_deliveries']

X = df[feature_cols + ['Time_taken(min)']].dropna()
y = X['Time_taken(min)']
X = X[feature_cols]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(f"Train MAE: {mean_absolute_error(y_train, train_pred):.2f}")
print(f"Test MAE: {mean_absolute_error(y_test, test_pred):.2f}")
print(f"Train R2: {r2_score(y_train, train_pred):.3f}")
print(f"Test R2: {r2_score(y_test, test_pred):.3f}")

# Check predictions
print(f"\nSample predictions: {test_pred[:10]}")
print(f"Unique predictions: {len(np.unique(test_pred))}")
print(f"Mean prediction: {test_pred.mean():.2f}")
