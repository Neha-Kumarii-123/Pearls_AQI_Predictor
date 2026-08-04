import os
import hopsworks
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

load_dotenv()

def fetch_historical_data():
    # 1. Connect to Hopsworks Feature Store
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    # 2. Read directly from Feature Group v2
    print("Fetching historical features and targets from Feature Group 'karachi_aqi_features' v2...")
    fg = fs.get_feature_group(name="karachi_aqi_features", version=2)
    df = fg.read()

    print(f" Successfully fetched historical data! Shape: {df.shape}")
    print("Available columns:", df.columns.tolist())
    
    # Target values check karne ke liye lines
    print(" Unique Target AQI values sample:", df['target_aqi'].nunique())
    print(" Target AQI Summary:\n", df['target_aqi'].describe())
    print("\n--- First 10 Rows ---")
    print(df.head(10))
    
    return df


def prepare_training_data(df):
    """
    Separate the regression target from the feature matrix and scale only the
    feature columns. `target_aqi` is preserved as the direct regression label.
    """
    feature_cols = [
        col for col in df.columns
        if col not in {"city", "timestamp", "target_aqi"}
    ]
    X = df[feature_cols]
    y = df["target_aqi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_baseline_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
    }
    print("📈 Baseline model evaluation:", metrics)
    return model, metrics


if __name__ == "__main__":
    df = fetch_historical_data()
    X_train, X_test, y_train, y_test = prepare_training_data(df)
    train_baseline_model(X_train, X_test, y_train, y_test)