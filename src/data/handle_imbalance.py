
import pandas as pd

from pathlib import Path
from imblearn.over_sampling import SMOTE


PROCESSED_DIR = Path("dataset/processed")

RANDOM_STATE = 42


def load_data():

    print("Loading selected features...")

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train_selected.csv"
    )

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test_selected.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).squeeze("columns")

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).squeeze("columns")

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):

    print("\nClass distribution before SMOTE:")
    print(y_train.value_counts())

    print("\nApplying SMOTE...")

    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=5
    )

    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nClass distribution after SMOTE:")
    print(y_train_balanced.value_counts())

    return X_train_balanced, y_train_balanced


def save_data(X_train_balanced, y_train_balanced):

    print("\nSaving balanced training data...")

    X_train_balanced.to_csv(
        PROCESSED_DIR / "X_train_balanced.csv",
        index=False
    )

    y_train_balanced.to_csv(
        PROCESSED_DIR / "y_train_balanced.csv",
        index=False
    )

    print("Balanced training data saved.")


def main():

    X_train, X_test, y_train, y_test = load_data()

    X_train_balanced, y_train_balanced = apply_smote(
        X_train,
        y_train
    )

    save_data(
        X_train_balanced,
        y_train_balanced
    )

    print("\n========== SMOTE COMPLETE ==========")

    print("Balanced training shape:", X_train_balanced.shape)
    print("Original testing shape:", X_test.shape)

    print("\nTesting class distribution (unchanged):")
    print(y_test.value_counts())


if __name__ == "__main__":
    main()
