import os
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

load_dotenv()

def main():
    print("--- Connecting to Hopsworks Feature Store ---")
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    print("--- Fetching Feature Group ---")
    feature_group = fs.get_feature_group(name="karachi_aqi_features", version=3)
    df = feature_group.read()
    
    print(f"Data loaded successfully with shape: {df.shape}")
    
    target_col = 'target_aqi' if 'target_aqi' in df.columns else 'target'
    
    # Creating Shifted Targets for Day+1, Day+2, Day+3
    df['target_day1'] = df[target_col].shift(-24)
    df['target_day2'] = df[target_col].shift(-48)
    df['target_day3'] = df[target_col].shift(-72)
    
    eval_df = df.dropna(subset=['target_day1', 'target_day2', 'target_day3']).copy()
    
    drop_cols = ['city', 'timestamp', target_col, 'target_day1', 'target_day2', 'target_day3']
    X_clean = eval_df.drop(columns=[col for col in drop_cols if col in eval_df.columns], errors='ignore')
    
    y_d1 = eval_df['target_day1']
    y_d2 = eval_df['target_day2']
    y_d3 = eval_df['target_day3']
    
    X_train, X_test, y1_train, y1_test = train_test_split(X_clean, y_d1, test_size=0.2, shuffle=False)
    _, _, y2_train, y2_test = train_test_split(X_clean, y_d2, test_size=0.2, shuffle=False)
    _, _, y3_train, y3_test = train_test_split(X_clean, y_d3, test_size=0.2, shuffle=False)
    
    print("--- Training Tuned LightGBM Models for 3 Horizons ---")
    
    # Tuned Hyperparameters
    tuned_params = {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    model_d1 = lgb.LGBMRegressor(**tuned_params)
    model_d2 = lgb.LGBMRegressor(**tuned_params)
    model_d3 = lgb.LGBMRegressor(**tuned_params)
    
    model_d1.fit(X_train, y1_train)
    model_d2.fit(X_train, y2_train)
    model_d3.fit(X_train, y3_train)
    
    preds_d1 = model_d1.predict(X_test)
    preds_d2 = model_d2.predict(X_test)
    preds_d3 = model_d3.predict(X_test)
    
    def print_metrics(horizon_name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"\n📊 [{horizon_name} Evaluation - Tuned LGBM]")
        print(f"  - MAE : {mae:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - R²  : {r2:.4f}")

    print_metrics("Day +1 Horizon", y1_test, preds_d1)
    print_metrics("Day +2 Horizon", y2_test, preds_d2)
    print_metrics("Day +3 Horizon", y3_test, preds_d3)

if __name__ == "__main__":
    main()