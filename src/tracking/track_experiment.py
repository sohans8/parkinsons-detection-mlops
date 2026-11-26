
import mlflow
import pandas as pd
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


PROCESSED_DIR = Path("dataset/processed")
ARTIFACT_DIR = Path("artifacts")
EVALUATION_DIR = Path("reports/model_evaluation")
EXPLAINABILITY_DIR = Path("reports/explainability")
COMPARISON_DIR = Path("reports/model_comparison")

EXPERIMENT_NAME = "Parkinsons_Detection_XGBoost"


def load_data():

    print("Loading test dataset...")

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv"
    )

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).squeeze("columns")

    print("Test data shape:", X_test.shape)

    return X_test, y_test


def load_model():

    print("\nLoading final trained model...")

    model = joblib.load(
        ARTIFACT_DIR / "xgboost_pipeline.joblib"
    )

    print("Model loaded successfully.")

    return model


def evaluate_model(model, X_test, y_test):

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    print("\nEvaluation metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


def track_experiment(model, X_test, metrics):

    print("\nStarting MLflow experiment tracking...")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="XGBoost_Final_Evaluation"
    ):

        # Log model parameters
        model_params = {
            "model_name": "XGBoost",
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "feature_selection_method": "Random Forest",
            "selected_features": 100,
            "scaling": "StandardScaler",
            "imbalance_handling": "SMOTE",
            "cross_validation": "StratifiedGroupKFold",
            "cv_splits": 5,
            "test_samples": len(X_test),
            "input_features": X_test.shape[1]
        }

        mlflow.log_params(model_params)

        # Log evaluation metrics
        mlflow.log_metrics(metrics)

        # Log trained model and metadata
        mlflow.log_artifact(
            str(ARTIFACT_DIR / "xgboost_pipeline.joblib")
        )

        mlflow.log_artifact(
            str(ARTIFACT_DIR / "model_metadata.json")
        )

        # Log evaluation reports
        if EVALUATION_DIR.exists():
            mlflow.log_artifacts(
                str(EVALUATION_DIR),
                artifact_path="model_evaluation"
            )

        # Log SHAP explainability reports
        if EXPLAINABILITY_DIR.exists():
            mlflow.log_artifacts(
                str(EXPLAINABILITY_DIR),
                artifact_path="explainability"
            )

        # Log model comparison results
        comparison_file = COMPARISON_DIR / "model_comparison.csv"

        if comparison_file.exists():
            mlflow.log_artifact(
                str(comparison_file),
                artifact_path="model_comparison"
            )

        print("\nMLflow tracking completed.")

        print("Run ID:", mlflow.active_run().info.run_id)

        print("Experiment:", EXPERIMENT_NAME)


def main():

    X_test, y_test = load_data()

    model = load_model()

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    track_experiment(
        model,
        X_test,
        metrics
    )

    print("\n========== MLFLOW TRACKING COMPLETE ==========")


if __name__ == "__main__":
    main()
