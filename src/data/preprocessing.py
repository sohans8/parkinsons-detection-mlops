
import json
import pandas as pd

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


RAW_DATA_PATH = Path("dataset/raw/pd_speech_features.csv")
PROCESSED_DIR = Path("dataset/processed")

TARGET_COLUMN = "class"
GROUP_COLUMN = "id"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_dataset():

    print("Loading raw dataset...")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH, header=1)

    print("Original dataset shape:", df.shape)

    return df


def validate_dataset(df):

    print("\nValidating dataset...")

    required_columns = ["id", "gender", "class"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.isnull().any().any():
        raise ValueError(
            "Dataset contains missing values."
        )

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    if df[GROUP_COLUMN].isnull().any():
        raise ValueError("Participant IDs cannot be missing.")

    print("Dataset validation successful.")


def remove_duplicates(df):

    print("\nRemoving exact duplicate records...")

    original_count = len(df)

    df = df.drop_duplicates().copy()

    removed_count = original_count - len(df)

    print("Original recordings:", original_count)
    print("Duplicates removed:", removed_count)
    print("Remaining recordings:", len(df))

    return df


def split_dataset(df):

    print("\nPreparing features and target...")

    # Exclude participant ID and target.
    # Gender is retained as an input variable.
    feature_columns = [
        col for col in df.columns
        if col not in [GROUP_COLUMN, TARGET_COLUMN]
    ]

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()
    groups = df[GROUP_COLUMN].copy()

    print("Input features:", X.shape[1])
    print("Target distribution:")
    print(y.value_counts())

    print("\nPerforming participant-aware split...")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()

    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()

    groups_train = groups.iloc[train_indices].copy()
    groups_test = groups.iloc[test_indices].copy()

    # Verify there is no participant overlap.
    overlap = set(groups_train) & set(groups_test)

    if overlap:
        raise RuntimeError(
            "Participant leakage detected!"
        )

    print("\nSplit completed.")

    print("Training recordings:", len(X_train))
    print("Testing recordings:", len(X_test))

    print("Training participants:", groups_train.nunique())
    print("Testing participants:", groups_test.nunique())

    print("Participant overlap:", len(overlap))

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        feature_columns
    )


def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    groups_train,
    groups_test,
    feature_columns
):

    print("\nSaving processed datasets...")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_train.to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False
    )

    y_train.to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False
    )

    groups_train.to_csv(
        PROCESSED_DIR / "groups_train.csv",
        index=False
    )

    groups_test.to_csv(
        PROCESSED_DIR / "groups_test.csv",
        index=False
    )

    metadata = {
        "original_feature_count": len(feature_columns),
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "duplicate_removal": "Exact duplicate rows removed",
        "split_strategy": "GroupShuffleSplit",
        "gender_included": True
    }

    with open(
        PROCESSED_DIR / "metadata.json",
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("Processed datasets saved successfully.")


def main():

    df = load_dataset()

    validate_dataset(df)

    df = remove_duplicates(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        feature_columns
    ) = split_dataset(df)

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        feature_columns
    )

    print("\n========== PREPROCESSING COMPLETE ==========")


if __name__ == "__main__":
    main()
