
import json
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


PROCESSED_DIR = Path("dataset/processed")
OUTPUT_DIR = Path("reports/feature_selection")

TOP_FEATURES = 100
RANDOM_STATE = 42


def load_data():

    print("Loading preprocessed datasets...")

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    return X_train, X_test, y_train


def train_feature_selector(X_train, y_train):

    print("\nTraining Random Forest for feature selection...")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Feature selection model trained.")

    return model


def select_features(model, X_train, X_test):

    print("\nRanking features by importance...")

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    selected_features = importance_df.head(TOP_FEATURES)["Feature"].tolist()

    X_train_selected = X_train[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()

    print("\nFeature selection completed.")

    print("Original feature count:", X_train.shape[1])
    print("Selected feature count:", len(selected_features))

    return (
        X_train_selected,
        X_test_selected,
        selected_features,
        importance_df
    )


def save_results(
    X_train_selected,
    X_test_selected,
    selected_features,
    importance_df
):

    print("\nSaving selected features...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train_selected.to_csv(
        PROCESSED_DIR / "X_train_selected.csv",
        index=False
    )

    X_test_selected.to_csv(
        PROCESSED_DIR / "X_test_selected.csv",
        index=False
    )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "selected_features.json",
        "w"
    ) as file:

        json.dump(
            selected_features,
            file,
            indent=4
        )

    print("Selected feature datasets saved.")
    print("Feature importance report saved.")


def main():

    X_train, X_test, y_train = load_data()

    model = train_feature_selector(
        X_train,
        y_train
    )

    (
        X_train_selected,
        X_test_selected,
        selected_features,
        importance_df
    ) = select_features(
        model,
        X_train,
        X_test
    )

    save_results(
        X_train_selected,
        X_test_selected,
        selected_features,
        importance_df
    )

    print("\n========== FEATURE SELECTION COMPLETE ==========")


if __name__ == "__main__":
    main()
