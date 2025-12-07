"""
Q2.3: Slow Inference Optimization
This prediction API takes 5 seconds per request. Profile and optimize it.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import time

# Train a model first (run once)
def train_model():
    df = pd.read_csv('/mnt/user-data/uploads/uber-eats-deliveries.csv')
    df['Time_taken(min)'] = df['Time_taken(min)'].str.replace('(min) ', '').astype(float)
    
    df['delivery_distance'] = np.sqrt(
        (df['Delivery_location_latitude'] - df['Restaurant_latitude'])**2 + 
        (df['Delivery_location_longitude'] - df['Restaurant_longitude'])**2
    ) * 111
    
    df['hour'] = pd.to_datetime(df['Time_Orderd']).dt.hour
    df['Weather_encoded'] = df['Weatherconditions'].str.replace('conditions ', '').astype('category').cat.codes
    df['Traffic_encoded'] = df['Road_traffic_density'].astype('category').cat.codes
    df['Vehicle_encoded'] = df['Type_of_vehicle'].astype('category').cat.codes
    
    feature_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 
                    'delivery_distance', 'hour', 'Weather_encoded', 
                    'Traffic_encoded', 'Vehicle_encoded', 'Vehicle_condition']
    
    X = df[feature_cols].dropna()
    y = df.loc[X.index, 'Time_taken(min)']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    with open('/tmp/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved!")


# Prediction API function (this is slow!)
def predict_delivery_time(delivery_data):
    """
    Predicts delivery time for a single delivery request.
    
    Args:
        delivery_data: dict with keys:
            - restaurant_lat, restaurant_lon
            - delivery_lat, delivery_lon
            - person_age, person_rating
            - weather, traffic, vehicle_type, vehicle_condition
            - order_time (format: "HH:MM:SS")
    
    Returns:
        Predicted delivery time in minutes
    """
    
    # Load model from disk every time
    with open('/tmp/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Calculate distance inefficiently
    distance = 0
    lat_diff = delivery_data['delivery_lat'] - delivery_data['restaurant_lat']
    lon_diff = delivery_data['delivery_lon'] - delivery_data['restaurant_lon']
    
    # Simulate expensive calculation
    for i in range(1000):
        distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111
    
    # Extract hour inefficiently
    hour = int(delivery_data['order_time'].split(':')[0])
    
    # Encode categoricals by loading reference data
    df_ref = pd.read_csv('/mnt/user-data/uploads/uber-eats-deliveries.csv')
    
    weather_map = {w.replace('conditions ', ''): i for i, w in 
                   enumerate(df_ref['Weatherconditions'].unique())}
    traffic_map = {t: i for i, t in enumerate(df_ref['Road_traffic_density'].unique())}
    vehicle_map = {v: i for i, v in enumerate(df_ref['Type_of_vehicle'].unique())}
    
    weather_encoded = weather_map.get(delivery_data['weather'], 0)
    traffic_encoded = traffic_map.get(delivery_data['traffic'], 0)
    vehicle_encoded = vehicle_map.get(delivery_data['vehicle_type'], 0)
    
    # Create feature array
    features = np.array([[
        delivery_data['person_age'],
        delivery_data['person_rating'],
        distance,
        hour,
        weather_encoded,
        traffic_encoded,
        vehicle_encoded,
        delivery_data['vehicle_condition']
    ]])
    
    # Predict
    prediction = model.predict(features)[0]
    
    return prediction


# Test the API
if __name__ == "__main__":
    # Train model once
    train_model()
    
    # Test prediction speed
    test_data = {
        'restaurant_lat': 22.745049,
        'restaurant_lon': 75.892471,
        'delivery_lat': 22.765049,
        'delivery_lon': 75.912471,
        'person_age': 35,
        'person_rating': 4.8,
        'weather': 'Sunny',
        'traffic': 'Medium',
        'vehicle_type': 'motorcycle',
        'vehicle_condition': 1,
        'order_time': '18:30:00'
    }
    
    # Time multiple predictions
    print("\nTesting prediction speed...")
    times = []
    for i in range(5):
        start = time.time()
        pred = predict_delivery_time(test_data)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Request {i+1}: {elapsed:.3f}s - Predicted: {pred:.1f} min")
    
    print(f"\nAverage time per request: {np.mean(times):.3f}s")
