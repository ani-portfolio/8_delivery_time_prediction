"""
Q2.4: Low Model Accuracy
Model accuracy is only 50%. Investigate feature distributions, 
check for data quality issues, and improve preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/mnt/user-data/uploads/uber-eats-deliveries.csv')

print("="*60)
print("INITIAL DATA EXPLORATION")
print("="*60)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Clean target variable (with a bug!)
df['Time_taken(min)'] = df['Time_taken(min)'].str.replace('(min) ', '')
# BUG: Not converting to numeric properly for some values

# Feature engineering
df['delivery_distance'] = (df['Delivery_location_latitude'] - df['Restaurant_latitude']) + \
                          (df['Delivery_location_longitude'] - df['Restaurant_longitude'])
# BUG: Wrong distance calculation (should be Euclidean)

df['hour'] = pd.to_datetime(df['Time_Orderd'], errors='coerce').dt.hour

# Encode categorical variables without proper handling
df['Weather_encoded'] = pd.factorize(df['Weatherconditions'])[0]
df['Traffic_encoded'] = pd.factorize(df['Road_traffic_density'])[0]  
df['Vehicle_encoded'] = pd.factorize(df['Type_of_vehicle'])[0]

# BUG: Not handling missing values in multiple_deliveries
# BUG: Not stripping whitespace from categorical columns

# Select features
feature_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 
                'delivery_distance', 'hour',
                'Weather_encoded', 'Traffic_encoded', 'Vehicle_encoded',
                'Vehicle_condition', 'multiple_deliveries']

# Prepare X and y (without proper data quality checks)
X = df[feature_cols]
y = df['Time_taken(min)']

# BUG: Not checking for NaNs, not converting y to numeric
print(f"\n{X.isnull().sum()}")
print(f"\nTarget variable type: {y.dtype}")

# Simple train-test split without checking data quality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
# BUG: Model hyperparameters not tuned, max_depth too shallow

try:
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate (may error out due to data issues)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTrain MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Train R2: {train_r2:.3f}")
    print(f"Test R2: {test_r2:.3f}")
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
except Exception as e:
    print(f"\nError during training/evaluation: {e}")
    print("\nDebug this! Check:")
    print("- Data types of features and target")
    print("- Missing values")
    print("- Feature engineering correctness")
    print("- Categorical encoding")
