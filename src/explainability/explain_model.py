
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


PROCESSED_DIR = Path("dataset/processed")
ARTIFACT_DIR = Path("artifacts")
OUTPUT_DIR = Path("reports/explainability")


def load_data():

    print("Loading test data...")

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv"
    )

    print("Test data shape:", X_test.shape)

    return X_test


def load_model():

    print("\nLoading trained XGBoost pipeline...")

    model = joblib.load(
        ARTIFACT_DIR / "xgboost_pipeline.joblib"
    )

    print("Model loaded successfully.")

    return model


def prepare_test_features(model, X_test):

    print("\nPreparing test features for SHAP...")

    feature_selector = model.named_steps["feature_selection"]
    scaler = model.named_steps["scaling"]

    selected_mask = feature_selector.get_support()

    selected_features = X_test.columns[selected_mask].tolist()

    X_selected = feature_selector.transform(X_test)

    X_scaled = scaler.transform(X_selected)

    X_transformed = pd.DataFrame(
        X_scaled,
        columns=selected_features
    )

    print("Original features:", X_test.shape[1])
    print("Selected features:", X_transformed.shape[1])

    return X_transformed


def calculate_shap_values(model, X_transformed):

    print("\nCalculating SHAP values...")

    xgb_model = model.named_steps["classifier"]

    explainer = shap.TreeExplainer(xgb_model)

    shap_values = explainer.shap_values(
        X_transformed
    )

    # Handle different SHAP output formats.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    print("SHAP values calculated successfully.")

    return shap_values


def save_feature_importance(shap_values, X_transformed):

    print("\nCalculating SHAP feature importance...")

    importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": X_transformed.columns,
        "Mean_Absolute_SHAP": importance
    })

    importance_df = importance_df.sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False
    )

    importance_df.to_csv(
        OUTPUT_DIR / "shap_feature_importance.csv",
        index=False
    )

    print("\nTop 10 important features:")

    print(
        importance_df.head(10).to_string(index=False)
    )


def save_shap_plots(shap_values, X_transformed):

    print("\nGenerating SHAP summary plots...")

    # SHAP beeswarm plot
    shap.summary_plot(
        shap_values,
        X_transformed,
        show=False,
        max_display=20
    )

    plt.title("SHAP Feature Impact Summary")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # SHAP bar plot
    shap.summary_plot(
        shap_values,
        X_transformed,
        plot_type="bar",
        show=False,
        max_display=20
    )

    plt.title("Top Features by SHAP Importance")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_bar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("SHAP plots saved successfully.")


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_test = load_data()

    model = load_model()

    X_transformed = prepare_test_features(
        model,
        X_test
    )

    shap_values = calculate_shap_values(
        model,
        X_transformed
    )

    save_feature_importance(
        shap_values,
        X_transformed
    )

    save_shap_plots(
        shap_values,
        X_transformed
    )

    print("\n========== SHAP EXPLAINABILITY COMPLETE ==========")


if __name__ == "__main__":
    main()
