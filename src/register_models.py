"""Register the final AQI local artifacts in the Hopsworks Model Registry.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import hopsworks
from dotenv import load_dotenv

from feature_engineering import MODEL_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = REPO_ROOT / ".env"

MODEL_SPECS = [
    {
        "name": "karachi_aqi_day1_xgboost",
        "artifact": "karachi_aqi_day1_xgboost_v6.pkl",
        "description": (
            "Final XGBoost model for Karachi AQI Day +1 (24-hour ahead) "
            "prediction using the canonical 100-feature contract from "
            "Hopsworks Feature Group v6."
        ),
        "horizon": "Day +1",
        "target_column": "target_day1",
        "metrics": {
            "mae": 7.6918,
            "rmse": 10.9789,
            "r2": 0.4843,
        },
        "selected_features": None,
        "selected_feature_file": None,
    },
    {
        "name": "karachi_aqi_day2_ridge",
        "artifact": "karachi_aqi_day2_ridge_v6.pkl",
        "feature_artifact": "karachi_aqi_day2_features_v6.pkl",
        "description": (
            "Final Ridge Regression model for Karachi AQI Day +2 "
            "(48-hour ahead) prediction using the canonical 100-feature "
            "contract from Hopsworks Feature Group v6 and a 60-feature "
            "selection."
        ),
        "horizon": "Day +2",
        "target_column": "target_day2",
        "metrics": {
            "mae": 9.8786,
            "rmse": 13.7769,
            "r2": 0.1889,
        },
        "selected_features": None,
        "selected_feature_file": "karachi_aqi_day2_features_v6.pkl",
    },
    {
        "name": "karachi_aqi_day3_ridge",
        "artifact": "karachi_aqi_day3_ridge_v6.pkl",
        "feature_artifact": "karachi_aqi_day3_features_v6.pkl",
        "description": (
            "Final Ridge Regression model for Karachi AQI Day +3 "
            "(72-hour ahead) prediction using the canonical 100-feature "
            "contract from Hopsworks Feature Group v6 and a 60-feature "
            "selection."
        ),
        "horizon": "Day +3",
        "target_column": "target_day3",
        "metrics": {
            "mae": 10.4410,
            "rmse": 14.7406,
            "r2": 0.0728,
        },
        "selected_features": None,
        "selected_feature_file": "karachi_aqi_day3_features_v6.pkl",
    },
]

def load_environment() -> None:
    if DOTENV_PATH.exists():
        load_dotenv(dotenv_path=str(DOTENV_PATH), override=False)
    else:
        load_dotenv(override=False)


def authenticate_project() -> Any:
    load_environment()
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY is not set.")
    return hopsworks.login(api_key_value=api_key, host="eu-west.cloud.hopsworks.ai")


def inspect_registry() -> list[dict[str, Any]]:
    project = authenticate_project()
    registry = project.get_model_registry()
    plan: list[dict[str, Any]] = []

    for spec in MODEL_SPECS:
        versions = sorted(m.version for m in registry.get_models(spec["name"]))
        next_version = max(versions) + 1 if versions else 1
        plan.append(
            {
                "name": spec["name"],
                "existing_versions": versions,
                "next_version": next_version,
                "description": spec["description"],
                "metrics": spec["metrics"],
            }
        )

    return plan


def load_selected_features(spec: dict[str, Any]) -> list[str] | None:
    feature_file = spec.get("selected_feature_file")
    if not feature_file:
        return None

    file_path = REPO_ROOT / feature_file
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing selected-feature artifact for {spec['name']}: {file_path}"
        )

    features = joblib.load(file_path)
    if not isinstance(features, list):
        raise TypeError(f"Expected a list for {feature_file}, found {type(features)!r}")
    return features


def build_model_bundle(spec: dict[str, Any], *, project_name: str | None = None) -> tuple[Path, dict[str, Any]]:
    model_artifact = REPO_ROOT / spec["artifact"]
    if not model_artifact.exists():
        raise FileNotFoundError(f"Missing model artifact for {spec['name']}: {model_artifact}")

    selected_features = load_selected_features(spec)
    metadata = {
        "model_name": spec["name"],
        "horizon": spec["horizon"],
        "target_column": spec["target_column"],
        "training_metrics": spec["metrics"],
        "canonical_feature_count": len(MODEL_FEATURES),
        "canonical_model_features": list(MODEL_FEATURES),
        "selected_feature_count": len(selected_features) if selected_features else None,
        "selected_features": selected_features,
        "required_raw_columns": [
            "timestamp",
            "pm25",
            "pm10",
            "ozone",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "carbon_monoxide",
            "temperature",
            "humidity",
            "target_aqi",
        ],
        "feature_group": "karachi_aqi_features",
        "feature_group_version": 6,
        "project_name": project_name,
    }

    temporary_dir = Path(tempfile.mkdtemp(prefix=f"{spec['name']}_"))
    bundled_model = temporary_dir / spec["artifact"]
    shutil.copy2(model_artifact, bundled_model)

    if spec.get("selected_feature_file"):
        feature_copy = temporary_dir / spec["selected_feature_file"]
        shutil.copy2(REPO_ROOT / spec["selected_feature_file"], feature_copy)

    joblib.dump(metadata, temporary_dir / "model_metadata.pkl")
    return temporary_dir, metadata


def print_registry_plan(plan: list[dict[str, Any]]) -> None:
    print("\nHopsworks Model Registry inspection")
    print("=" * 72)
    for item in plan:
        existing = item["existing_versions"]
        next_version = item["next_version"]
        print(
            f"- {item['name']}: existing versions={existing or 'NONE'}; "
            f"new version to be created={next_version}"
        )
    print("\nThe three target artifacts are:")
    for item in plan:
        print(f"  * {item['name']} -> {item['metrics']}")
    print("\nNo registration was executed in this review pass.")


def register_model(project: Any, spec: dict[str, Any]) -> int:
    registry = project.get_model_registry()

    versions = [
        m.version
        for m in registry.get_models(spec["name"])
    ]

    next_version = max(versions) + 1 if versions else 1

    bundle_dir, metadata = build_model_bundle(
        spec,
        project_name=project.project_namespace
    )

    try:
        model = registry.python.create_model(
            name=spec["name"],
            version=next_version,
            metrics=spec["metrics"],
            description=spec["description"],
        )

        model.save(str(bundle_dir))

        print(
            f"Registered {spec['name']} as version {model.version} "
            f"with metrics={spec['metrics']}"
        )

        return model.version

    finally:
        shutil.rmtree(
            bundle_dir,
            ignore_errors=True
        )

def register_all(project: Any, *, apply: bool) -> None:
    plan = inspect_registry()
    if not apply:
        print_registry_plan(plan)
        return

    print("\nApplying the planned Hopsworks registrations.")
    for spec in MODEL_SPECS:
        model_name = spec["name"]
        next_version = max((m.version for m in project.get_model_registry().get_models(model_name)), default=0) + 1
        print(f"- {model_name}: registering version {next_version}")
    print()

    for spec in MODEL_SPECS:
        existing_versions = [
            m.version
            for m in project.get_model_registry().get_models(spec["name"])
        ]

        
        register_model(project, spec)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Review or apply Hopsworks Model Registry registration for the final "
            "Day +1/+2/+3 AQI models."
        )
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Actually register the models. Default is a dry-run review only.",
    )
    args = parser.parse_args()

    project = authenticate_project()
    plan = inspect_registry()
    print_registry_plan(plan)

    if args.register:
        print("\nRegistration was explicitly requested; proceeding to save the models.")
        register_all(project, apply=True)

    if not args.register:
        print("\nDry-run complete. Review the plan above and rerun with --register to apply it.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted before registration.", file=sys.stderr)
        raise SystemExit(130)
