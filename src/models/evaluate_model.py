
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


PROCESSED_DIR = Path("dataset/processed")
ARTIFACT_DIR = Path("artifacts")
REPORT_DIR = Path("reports/model_evaluation")


def load_test_data():

    print("Loading untouched test data...")

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv"
    )

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).squeeze("columns")

    print("Test data shape:", X_test.shape)
    print("Test labels:", y_test.shape)

    return X_test, y_test


def load_model():

    model_path = ARTIFACT_DIR / "xgboost_pipeline.joblib"

    print("\nLoading trained XGBoost model...")

    model = joblib.load(model_path)

    print("Model loaded successfully.")

    return model


def evaluate_model(model, X_test, y_test):

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_prob)
    }

    print("\n========== FINAL TEST RESULTS ==========")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Healthy", "Parkinson's"],
        zero_division=0
    )

    print(report)

    return y_pred, y_prob, metrics, report


def save_evaluation_results(
    y_test,
    y_pred,
    y_prob,
    metrics,
    report
):

    print("\nSaving evaluation results...")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save metrics
    with open(REPORT_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    # Save classification report
    with open(REPORT_DIR / "classification_report.txt", "w") as file:
        file.write(report)

    # Save predictions
    predictions = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred,
        "Parkinson_Probability": y_prob
    })

    predictions.to_csv(
        REPORT_DIR / "predictions.csv",
        index=False
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Healthy", "Parkinson's"]
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title("XGBoost Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ROC curve
    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
        name="XGBoost"
    )

    plt.title("XGBoost ROC Curve")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close()

    print("Evaluation reports saved to:", REPORT_DIR)


def main():

    X_test, y_test = load_test_data()

    model = load_model()

    y_pred, y_prob, metrics, report = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_evaluation_results(
        y_test,
        y_pred,
        y_prob,
        metrics,
        report
    )

    print("\n========== MODEL EVALUATION COMPLETE ==========")


if __name__ == "__main__":
    main()
