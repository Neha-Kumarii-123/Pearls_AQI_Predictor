import os
from dotenv import load_dotenv
import hopsworks
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from feature_engineering import (
    MODEL_FEATURES,
    REQUIRED_RAW_COLUMNS,
    build_rich_features,
    validate_feature_frame,
)

load_dotenv()


def main():

    print("--- Connecting to Hopsworks Feature Store ---")

    project = hopsworks.login()
    fs = project.get_feature_store()

    print("--- Fetching Feature Group ---")

    feature_group = fs.get_feature_group(
        name="karachi_aqi_features",
        version=4
    )

    df = feature_group.select(list(REQUIRED_RAW_COLUMNS)).read()

    print("Raw Hopsworks shape:", df.shape)
    features = build_rich_features(df)
    validate_feature_frame(features)
    print("Shared feature-frame shape:", features.shape)
    print("Number of MODEL_FEATURES:", len(MODEL_FEATURES))

    features["target_day2"] = features["target_aqi"].shift(-48)
    eval_df = features.dropna(
        subset=list(MODEL_FEATURES) + ["target_day2"]
    ).copy()

    X = eval_df[list(MODEL_FEATURES)]
    y = eval_df["target_day2"]
    print("Actual Ridge input feature count:", len(X.columns))
    print("Whether target_day2 is in X:", "target_day2" in X.columns)
    print("Whether target_aqi is in X:", "target_aqi" in X.columns)
    print("Whether X columns exactly match MODEL_FEATURES:", list(X.columns) == list(MODEL_FEATURES))

    if len(MODEL_FEATURES) != 100:
        raise RuntimeError("Expected exactly 100 MODEL_FEATURES")
    if len(X.columns) != 100:
        raise RuntimeError("Expected exactly 100 X columns")
    if list(X.columns) != list(MODEL_FEATURES):
        raise RuntimeError("X columns do not match canonical MODEL_FEATURES")
    if "target_day2" in X.columns or "target_aqi" in X.columns:
        raise RuntimeError("Forecast or direct target leaked into X")

    # ---------------------------------------------------------
    # Chronological 80/20 split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # ---------------------------------------------------------
    # Ridge Pipeline
    #
    # Scaling is important for Ridge because it is
    # sensitive to feature magnitudes.
    # ---------------------------------------------------------

    ridge_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=1500.0)
        )
    ])

    print("\n--- Training Final Ridge Model for Day +2 ---")

    ridge_model.fit(
        X_train,
        y_train
    )
        # ---------------------------------------------------------
    # Final Feature Selection
    # ---------------------------------------------------------
    # Based on our experiment, the top 60 Ridge-ranked features
    # performed best on the current validation split.
    # We now use those 60 features for the final Day +2 model.
    # ---------------------------------------------------------

    feature_coefficients = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": ridge_model.named_steps["ridge"].coef_
    })

    feature_coefficients["abs_coefficient"] = (
        feature_coefficients["coefficient"].abs()
    )

    feature_coefficients = feature_coefficients.sort_values(
        "abs_coefficient",
        ascending=False
    )

    selected_features = (
        feature_coefficients
        .head(60)["feature"]
        .tolist()
    )

    print("\n--- Final Feature Selection ---")
    print(f"Selected features: {len(selected_features)}")

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    final_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=1500.0)
        )
    ])

    print("\n--- Training Final 60-Feature Ridge Model for Day +2 ---")

    final_model.fit(
        X_train_selected,
        y_train
    )

    preds = final_model.predict(
        X_test_selected
    )

   
    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        preds
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds
        )
    )

    r2 = r2_score(
        y_test,
        preds
    )

    print(
        "\n📊 [Final Day +2 Ridge Model]"
    )

    print(
        f"  - MAE : {mae:.4f}"
    )

    print(
        f"  - RMSE: {rmse:.4f}"
    )

    print(
        f"  - R²  : {r2:.4f}"
    )

    # ---------------------------------------------------------
    # Persistence baseline
    # ---------------------------------------------------------

    baseline_preds = eval_df.loc[
        X_test.index,
        "target_aqi"
    ]

    base_mae = mean_absolute_error(
        y_test,
        baseline_preds
    )

    base_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_preds
        )
    )

    base_r2 = r2_score(
        y_test,
        baseline_preds
    )

    print(
        "\n📊 [Day +2 Persistence Baseline]"
    )

    print(
        f"  - MAE : {base_mae:.4f}"
    )

    print(
        f"  - RMSE: {base_rmse:.4f}"
    )

    print(
        f"  - R²  : {base_r2:.4f}"
    )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    print(
        "\n--- Final Model vs Persistence ---"
    )

    print(
        "MAE:",
        "BETTER" if mae < base_mae else "WORSE"
    )

    print(
        "RMSE:",
        "BETTER" if rmse < base_rmse else "WORSE"
    )

    print(
        "R²:",
        "BETTER" if r2 > base_r2 else "WORSE"
    )

    # # ---------------------------------------------------------
    # # Save model locally
    # # ---------------------------------------------------------

    model_file = "karachi_aqi_day2_ridge.pkl"

    joblib.dump(
        final_model,
        model_file
    )

    print(
        f"\nFinal model saved locally as: {model_file}"
    )

    features_file = "karachi_aqi_day2_features.pkl"

    joblib.dump(
        selected_features,
        features_file
    )

    print(
        f"Selected feature list saved as: {features_file}"
    )

if __name__ == "__main__":
    main()