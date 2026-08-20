import os
import hopsworks
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables from .env file
load_dotenv()


def fetch_data_for_eda():
    """Connects to Hopsworks and fetches the AQI feature dataset for Exploratory Data Analysis."""
    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    print("Fetching dataset from Feature Group 'karachi_aqi_features' v2...")
    fg = fs.get_feature_group(name="karachi_aqi_features", version=2)
    df = fg.read()
    return df


def analyze_outliers(df):
    """Detects outliers in numerical columns using the Interquartile Range (IQR) method."""
    print("\n--- Outlier Analysis (IQR Method) ---")
    
    # Select numerical columns excluding non-feature fields
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col not in {"timestamp"}]

    outlier_summary = {}

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        outlier_summary[col] = {
            "outlier_count": outlier_count,
            "percentage": round(outlier_percentage, 2)
        }
        print(f"Column '{col}': {outlier_count} outliers found ({outlier_percentage:.2f}%)")

    return outlier_summary


if __name__ == "__main__":
    # Fetch dataset
    df = fetch_data_for_eda()

    # Display basic dataset info
    print(f"\nDataset Shape: {df.shape}")
    print("\nMissing Values per Column:")
    print(df.isnull().sum())

    # Perform outlier detection
    outlier_summary = analyze_outliers(df)