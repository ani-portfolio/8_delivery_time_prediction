order_data_path_csv =  "/Users/ani/Projects/8_DELIVERY_TIME_PREDICTION/data/uber-eats-deliveries.csv"
results_path = "/Users/ani/Projects/8_DELIVERY_TIME_PREDICTION/results"
processed_train_data_path = "/Users/ani/Projects/8_DELIVERY_TIME_PREDICTION/results/processed_train_data.csv"
processed_test_data_path = "/Users/ani/Projects/8_DELIVERY_TIME_PREDICTION/results/processed_test_data.csv"
processed_full_data_path = "/Users/ani/Projects/8_DELIVERY_TIME_PREDICTION/results/processed_full_data.csv"
ARTIFACTS_DIR = "/Users/ani/Projects/8_delivery_time_prediction/src/artifacts"

raw_data_expected_columns = {
    'ID': 'object',
    'Delivery_person_ID': 'object',
    'Delivery_person_Age': 'object',
    'Delivery_person_Ratings': 'object',
    'Restaurant_latitude': 'float64',
    'Restaurant_longitude': 'float64',
    'Delivery_location_latitude': 'float64',
    'Delivery_location_longitude': 'float64',
    'Order_Date': 'object',
    'Time_Orderd': 'object',
    'Time_Order_picked': 'object',
    'Weatherconditions': 'object',
    'Road_traffic_density': 'object',
    'Vehicle_condition': 'int64',
    'Type_of_order': 'object',
    'Type_of_vehicle': 'object',
    'multiple_deliveries': 'object',
    'Festival': 'object',
    'City': 'object',
    'Time_taken(min)': 'object'
    }

mean_impute_columns = ['delivery_person_age']
median_impute_columns = ['delivery_person_ratings']
mode_impute_columns = ['time_orderd', 'road_traffic_density', 'multiple_deliveries', 'festival', 'city']

target = ['time_taken_min']

features = ['delivery_person_age',
 'vehicle_condition',
 'multiple_deliveries',
 'distance',
 'day_of_week',
 'hour_of_day',
 'driver_speed',
 'avg_driver_rating',
 'weatherconditions_conditionscloudy',
 'weatherconditions_conditionsfog',
 'weatherconditions_conditionsnan',
 'weatherconditions_conditionssandstorms',
 'weatherconditions_conditionsstormy',
 'weatherconditions_conditionssunny',
 'weatherconditions_conditionswindy',
 'road_traffic_density_high',
 'road_traffic_density_jam',
 'road_traffic_density_low',
 'road_traffic_density_medium',
 'type_of_order_buffet',
 'type_of_order_drinks',
 'type_of_order_meal',
 'type_of_order_snack',
 'type_of_vehicle_bicycle',
 'type_of_vehicle_electric_scooter',
 'type_of_vehicle_motorcycle',
 'type_of_vehicle_scooter',
 'festival_no',
 'festival_yes',
 'city_metropolitian',
 'city_semi_urban',
 'city_urban']

gbdt_params = {
    'fixed_params': {
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1
    },
    'search_space': {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 1000},
        'num_leaves': {'type': 'int', 'low': 2, 'high': 256},
        'max_depth': {'type': 'int', 'low': 2, 'high': 32},
        'min_child_samples': {'type': 'int', 'low': 20, 'high': 100},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0}
    }
}

gbdt_n_splits = 3
gbdt_n_trials = 5