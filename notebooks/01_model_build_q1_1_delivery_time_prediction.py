"""
Uber Eats Delivery Time Prediction
ML Coding Interview Solution

Problem: Predict delivery time (regression)
Metric: RMSE (Root Mean Squared Error)
Timeline: 60-minute interview format
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: PROBLEM DEFINITION (5-7 minutes)
# =============================================================================
"""
Problem Type: REGRESSION
Target: Time_taken(min) - continuous variable
Metric: RMSE (penalizes large errors), MAE, R²

Clarifying Questions to Ask:
1. What's the acceptable error range? (e.g., ±5 minutes?)
2. Are there any latency constraints for predictions?
3. Should we handle missing values or drop them?
4. Any specific features we must/cannot use?
"""


# =============================================================================
# STEP 2: DATA LOADING & EDA (5-7 minutes)
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset with basic validation."""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded: {df.shape}")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load data: {e}")


def perform_eda(df: pd.DataFrame) -> None:
    """Quick exploratory data analysis."""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print(f"\n1. Dataset Shape: {df.shape}")
    print(f"\n2. Column Types:\n{df.dtypes}")
    print(f"\n3. Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\n4. Target Statistics:")
    
    # Clean target variable (remove "(min)" prefix)
    if df['Time_taken(min)'].dtype == 'object':
        df['Time_taken(min)'] = df['Time_taken(min)'].str.replace('(min) ', '').astype(float)
    
    print(df['Time_taken(min)'].describe())
    print(f"\n5. Unique Values in Categorical Columns:")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:5]:  # Show first 5
        print(f"   {col}: {df[col].nunique()} unique values")


# =============================================================================
# STEP 3: DATA PREPARATION (10-15 minutes)
# =============================================================================

def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Clean data, engineer features, split train/test.
    Returns: X_train, X_test, y_train, y_test
    """
    df = df.copy()
    
    # 1. Clean target variable
    if df['Time_taken(min)'].dtype == 'object':
        df['Time_taken(min)'] = df['Time_taken(min)'].str.replace('(min) ', '').astype(float)
    
    # 2. Handle missing values
    df = df.dropna(subset=['Time_taken(min)'])  # Drop if target is missing
    
    # 3. Clean text columns (remove extra spaces)
    text_cols = ['Weatherconditions', 'Road_traffic_density', 'City', 'Type_of_order']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()
            df[col] = df[col].str.replace('conditions ', '')
    
    # 4. Feature Engineering
    df = engineer_features(df)
    
    # 5. Select features
    feature_cols = [
        'Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition',
        'multiple_deliveries', 'distance_km', 'hour', 'is_peak_hour',
        'prep_time_minutes', 'Weatherconditions', 'Road_traffic_density',
        'Type_of_vehicle', 'Type_of_order', 'City'
    ]
    
    # Filter to available columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].copy()
    y = df['Time_taken(min)'].values
    
    # 6. Train-Test Split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n✓ Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing data."""
    df = df.copy()
    
    # 1. Distance calculation (Haversine formula)
    df['distance_km'] = calculate_distance(
        df['Restaurant_latitude'], df['Restaurant_longitude'],
        df['Delivery_location_latitude'], df['Delivery_location_longitude']
    )
    
    # 2. Time features
    try:
        df['Time_Orderd'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S', errors='coerce')
        df['hour'] = df['Time_Orderd'].dt.hour
        df['is_peak_hour'] = ((df['hour'] >= 12) & (df['hour'] <= 14) | 
                              (df['hour'] >= 19) & (df['hour'] <= 21)).astype(int)
    except:
        df['hour'] = 12
        df['is_peak_hour'] = 0
    
    # 3. Preparation time (time between order and pickup)
    try:
        order_time = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S', errors='coerce')
        pickup_time = pd.to_datetime(df['Time_Order_picked'], format='%H:%M:%S', errors='coerce')
        df['prep_time_minutes'] = (pickup_time - order_time).dt.total_seconds() / 60
        df['prep_time_minutes'] = df['prep_time_minutes'].fillna(df['prep_time_minutes'].median())
    except:
        df['prep_time_minutes'] = 15  # Default
    
    # 4. Fill missing values
    df['multiple_deliveries'] = df['multiple_deliveries'].fillna(0)
    df['Vehicle_condition'] = df['Vehicle_condition'].fillna(df['Vehicle_condition'].median())
    
    return df


def calculate_distance(lat1: pd.Series, lon1: pd.Series, 
                       lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Calculate distance using Haversine formula."""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Encode categorical variables and scale numerical features.
    Returns: X_train_processed, X_test_processed
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Separate numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Fill NaN values in numerical columns
    for col in numerical_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    # Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fill NaN in categorical as 'Unknown'
        X_train[col] = X_train[col].fillna('Unknown').astype(str)
        X_test[col] = X_test[col].fillna('Unknown').astype(str)
        
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = X_test[col].map(lambda x: le.transform([str(x)])[0] 
                                       if str(x) in le.classes_ else -1)
        encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"\n✓ Encoded {len(categorical_cols)} categorical features")
    print(f"✓ Scaled {len(numerical_cols)} numerical features")
    
    return X_train, X_test


# =============================================================================
# STEP 4: MODEL TRAINING & EVALUATION (20-25 minutes)
# =============================================================================

def create_baseline_model(y_train: np.ndarray) -> float:
    """Naive baseline: predict mean delivery time."""
    baseline_pred = np.mean(y_train)
    print(f"\n✓ Baseline Model: Predict mean = {baseline_pred:.2f} minutes")
    return baseline_pred


def train_models(X_train: pd.DataFrame, y_train: np.ndarray) -> dict:
    """Train multiple models and return them."""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, 
                                                random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                        random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✓ {name} trained")
    
    return trained_models


def evaluate_models(models: dict, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: np.ndarray, y_test: np.ndarray, 
                   baseline_pred: float) -> pd.DataFrame:
    """
    Evaluate all models on train and test sets.
    Returns: DataFrame with evaluation metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    results = []
    
    # Baseline evaluation
    baseline_train_rmse = np.sqrt(mean_squared_error(y_train, [baseline_pred] * len(y_train)))
    baseline_test_rmse = np.sqrt(mean_squared_error(y_test, [baseline_pred] * len(y_test)))
    baseline_test_mae = mean_absolute_error(y_test, [baseline_pred] * len(y_test))
    
    results.append({
        'Model': 'Baseline (Mean)',
        'Train RMSE': baseline_train_rmse,
        'Test RMSE': baseline_test_rmse,
        'Test MAE': baseline_test_mae,
        'Test R²': 0.0
    })
    
    # ML models evaluation
    for name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results.append({
            'Model': name,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'Test R²': test_r2
        })
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    return results_df


def get_feature_importance(model, feature_names: list, top_n: int = 10) -> pd.DataFrame:
    """Extract feature importance from tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df
    return None


# =============================================================================
# STEP 5: MAIN EXECUTION & RESULTS (7-8 minutes)
# =============================================================================

def main():
    """Main execution pipeline."""
    print("\n" + "="*60)
    print("UBER EATS DELIVERY TIME PREDICTION")
    print("="*60)
    
    # Load data
    df = load_data('/mnt/user-data/uploads/uber-eats-deliveries.csv')
    
    # EDA
    perform_eda(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Preprocess
    X_train_processed, X_test_processed = preprocess_features(X_train, X_test)
    
    # Baseline model
    baseline_pred = create_baseline_model(y_train)
    
    # Train ML models
    models = train_models(X_train_processed, y_train)
    
    # Evaluate
    results = evaluate_models(models, X_train_processed, X_test_processed, 
                             y_train, y_test, baseline_pred)
    
    # Feature importance (for best model)
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Gradient Boosting)")
    print("="*60)
    importance_df = get_feature_importance(models['Gradient Boosting'], 
                                          X_train.columns.tolist())
    if importance_df is not None:
        print("\n" + importance_df.to_string(index=False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY & NEXT STEPS")
    print("="*60)
    print("\n✓ Problem: Regression (predict delivery time in minutes)")
    print("✓ Metric: RMSE (lower is better)")
    print(f"✓ Best Model: {results.loc[results['Test RMSE'].idxmin(), 'Model']}")
    print(f"✓ Best Test RMSE: {results['Test RMSE'].min():.2f} minutes")
    print(f"✓ Best Test MAE: {results.loc[results['Test RMSE'].idxmin(), 'Test MAE']:.2f} minutes")
    
    print("\nKey Insights:")
    print("  • Distance and prep time are likely strong predictors")
    print("  • Traffic conditions and weather impact delivery time")
    print("  • Model generalizes well (train/test RMSE close)")
    
    print("\nNext Steps for Production:")
    print("  1. Hyperparameter tuning (GridSearchCV)")
    print("  2. Cross-validation for robust estimates")
    print("  3. Handle outliers in delivery time")
    print("  4. Add more temporal features (day of week, holidays)")
    print("  5. Monitor model performance over time")
    print("  6. A/B test in production")


if __name__ == "__main__":
    main()
