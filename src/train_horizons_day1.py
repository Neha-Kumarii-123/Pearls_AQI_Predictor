import os
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

load_dotenv()

def main():
    print("--- Connecting to Hopsworks Feature Store ---")
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    print("--- Fetching Feature Group ---")
    feature_group = fs.get_feature_group(name="karachi_aqi_features", version=4)
    df = feature_group.read()
    
    # Sort and create features
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
        
    target_col = 'target_aqi' if 'target_aqi' in df.columns else 'target'
    
    # Feature Engineering (Lags + Rolling for Target & Weather Features)
    features_to_lag = [target_col, 'temperature', 'humidity', 'pm25', 'pm10', 'humidex']
    
    for col in features_to_lag:
        if col in df.columns:
            for lag in [1, 2, 3, 24, 48]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Rolling means & std to capture trends
            df[f'{col}_rolling_mean_24'] = df[col].shift(1).rolling(window=24).mean()
            df[f'{col}_rolling_std_24'] = df[col].shift(1).rolling(window=24).std()

    # Target shift for Day +1 prediction
    df['target_day1'] = df[target_col].shift(-24)
    df['target_day2'] = df[target_col].shift(-48)
    eval_df = df.dropna().copy()
    drop_cols = ['city', 'timestamp', target_col, 'target_day1','target_day2']
    X_clean = eval_df.drop(columns=[col for col in drop_cols if col in eval_df.columns], errors='ignore')
    print("Actual XGBoost input features:", len(X_clean.columns))
    y_d1 = eval_df['target_day1']
    y_d2 = eval_df['target_day2']
    
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_d1, test_size=0.2, shuffle=False)
    X_train_d2, X_test_d2, y_train_d2, y_test_d2 = train_test_split(
        X_clean, y_d2, test_size=0.2, shuffle=False
    )
    print("--- Training Optimized XGBoost Model for Day +1 ---")
    
    # Tuning XGBoost for better generalization
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,  # L1 regularization
        reg_lambda=1.0, # L2 regularization
        n_jobs=-1,
        random_state=42
    )
    
    xgb_model.fit(X_train, y_train)
    preds = xgb_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"\n📊 [Day +1 Horizon - Optimized XGBoost]")
    print(f"  - MAE : {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R²  : {r2:.4f}")
    # --- 2. Persistence Baseline Metrics ---
    # Persistence: use current AQI to predict AQI 24 hours ahead
    baseline_preds = eval_df.loc[X_test.index, target_col]

    base_mae = mean_absolute_error(y_test, baseline_preds)
    base_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    base_r2 = r2_score(y_test, baseline_preds)

    print(f"\n📊 [Persistence Baseline Model]")
    print(f"  - MAE : {base_mae:.4f}")
    print(f"  - RMSE: {base_rmse:.4f}")
    print(f"  - R²  : {base_r2:.4f}")

    # # --- Save Day +1 XGBoost Model Locally ---
    # model_file = "karachi_aqi_day1_xgboost.pkl"
    # joblib.dump(xgb_model, model_file)

    # print(f"\nModel saved locally as: {model_file}")

    # # --- Register Day +1 Model in Hopsworks Model Registry ---
    # print("\nRegistering Day +1 XGBoost model in Hopsworks Model Registry...")

    # mr = project.get_model_registry()

    # day1_model = mr.python.create_model(
    #     name="karachi_aqi_day1_xgboost",
    #     metrics={
    #         "mae": mae,
    #         "rmse": rmse,
    #         "r2": r2
    #     },
    #     description="Optimized XGBoost model for Karachi AQI Day +1 (24-hour ahead) prediction."
    # )

    # day1_model.save(model_file)

    # print("Day +1 XGBoost model successfully registered in Hopsworks Model Registry!")

        # --- Train Separate XGBoost Model for Day +2 ---
    print("\n--- Training Optimized XGBoost Model for Day +2 ---")

    xgb_model_d2 = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42
    )

    xgb_model_d2.fit(X_train_d2, y_train_d2)
    preds_d2 = xgb_model_d2.predict(X_test_d2)

    mae_d2 = mean_absolute_error(y_test_d2, preds_d2)
    rmse_d2 = np.sqrt(mean_squared_error(y_test_d2, preds_d2))
    r2_d2 = r2_score(y_test_d2, preds_d2)

    print(f"\n📊 [Day +2 Horizon - Optimized XGBoost]")
    print(f"  - MAE : {mae_d2:.4f}")
    print(f"  - RMSE: {rmse_d2:.4f}")
    print(f"  - R²  : {r2_d2:.4f}")

    # --- Day +2 Persistence Baseline ---
    baseline_preds_d2 = eval_df.loc[X_test_d2.index, target_col]

    base_mae_d2 = mean_absolute_error(y_test_d2, baseline_preds_d2)
    base_rmse_d2 = np.sqrt(mean_squared_error(y_test_d2, baseline_preds_d2))
    base_r2_d2 = r2_score(y_test_d2, baseline_preds_d2)

    print(f"\n📊 [Day +2 Persistence Baseline Model]")
    print(f"  - MAE : {base_mae_d2:.4f}")
    print(f"  - RMSE: {base_rmse_d2:.4f}")
    print(f"  - R²  : {base_r2_d2:.4f}")
if __name__ == "__main__":
    main()