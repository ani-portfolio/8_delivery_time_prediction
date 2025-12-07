"""
Q2.2: Memory Optimization
The training script runs out of memory with the full dataset. 
Optimize it to handle 10M rows efficiently.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load data - simulate large dataset by repeating
df = pd.read_csv('/mnt/user-data/uploads/uber-eats-deliveries.csv')
print(f"Original dataset size: {len(df)} rows")

# Simulate 10M rows (comment out for actual 10M dataset)
# df = pd.concat([df] * 1000, ignore_index=True)  
# print(f"Simulated dataset size: {len(df)} rows")

# Clean target
df['Time_taken(min)'] = df['Time_taken(min)'].str.replace('(min) ', '').astype(float)

# Feature engineering - creates multiple copies of data
df['delivery_distance'] = np.sqrt(
    (df['Delivery_location_latitude'] - df['Restaurant_latitude'])**2 + 
    (df['Delivery_location_longitude'] - df['Restaurant_longitude'])**2
) * 111

df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Time_Orderd'] = pd.to_datetime(df['Time_Orderd'])
df['Time_Order_picked'] = pd.to_datetime(df['Time_Order_picked'])

df['hour'] = df['Time_Orderd'].dt.hour
df['day_of_week'] = df['Order_Date'].dt.dayofweek
df['month'] = df['Order_Date'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_peak_hour'] = ((df['hour'] >= 12) & (df['hour'] <= 14)) | ((df['hour'] >= 19) & (df['hour'] <= 21))

# One-hot encode - creates many columns
df_encoded = pd.get_dummies(df, columns=['Weatherconditions', 'Road_traffic_density', 
                                          'Type_of_order', 'Type_of_vehicle', 
                                          'City', 'Festival'])

# Create polynomial features - explodes feature count
from sklearn.preprocessing import PolynomialFeatures
numeric_features = ['Delivery_person_Age', 'Delivery_person_Ratings', 
                    'delivery_distance', 'Vehicle_condition']

poly = PolynomialFeatures(degree=3, include_bias=False)
poly_features = poly.fit_transform(df[numeric_features].fillna(0))
poly_df = pd.DataFrame(poly_features, 
                       columns=[f'poly_{i}' for i in range(poly_features.shape[1])])

# Combine everything
df_final = pd.concat([df_encoded, poly_df], axis=1)

# Select features
feature_cols = [col for col in df_final.columns if col not in 
                ['ID', 'Delivery_person_ID', 'Order_Date', 'Time_Orderd', 
                 'Time_Order_picked', 'Time_taken(min)']]

X = df_final[feature_cols].fillna(0)
y = df['Time_taken(min)']

print(f"Feature matrix shape: {X.shape}")
print(f"Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model - loads everything into memory
print("\nTraining model...")
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nTest MAE: {mae:.2f}")
