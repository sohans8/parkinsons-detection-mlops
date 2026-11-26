
import json
import joblib

import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier


PROCESSED_DIR = Path("dataset/processed")
ARTIFACT_DIR = Path("artifacts")

RANDOM_STATE = 42
TOP_FEATURES = 100


def load_training_data():

    print("Loading original training data...")

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).squeeze("columns")

    print("Training data shape:", X_train.shape)
    print("Training labels:", y_train.shape)

    return X_train, y_train


def create_final_pipeline():

    feature_selector = SelectFromModel(
        estimator=RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        max_features=TOP_FEATURES,
        threshold=-float("inf")
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    pipeline = Pipeline([
        ("feature_selection", feature_selector),

        ("scaling", StandardScaler()),

        ("smote", SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=5
        )),

        ("classifier", xgb_model)
    ])

    return pipeline


def train_final_model(X_train, y_train):

    print("\nCreating final XGBoost pipeline...")

    model = create_final_pipeline()

    print("\nTraining final model...")

    model.fit(X_train, y_train)

    print("\nFinal model training completed.")

    return model


def save_model(model, X_train):

    print("\nSaving final model...")

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = ARTIFACT_DIR / "xgboost_pipeline.joblib"

    joblib.dump(
        model,
        model_path
    )

    selected_features = X_train.columns[
        model.named_steps["feature_selection"].get_support()
    ].tolist()

    metadata = {
        "model_name": "XGBoost",
        "input_features": len(X_train.columns),
        "selected_features_count": len(selected_features),
        "selected_features": selected_features,
        "random_state": RANDOM_STATE
    }

    metadata_path = ARTIFACT_DIR / "model_metadata.json"

    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)

    print("Model saved to:", model_path)
    print("Metadata saved to:", metadata_path)
    print("Selected features:", len(selected_features))


def main():

    X_train, y_train = load_training_data()

    model = train_final_model(
        X_train,
        y_train
    )

    save_model(
        model,
        X_train
    )

    print("\n========== FINAL MODEL TRAINING COMPLETE ==========")


if __name__ == "__main__":
    main()
