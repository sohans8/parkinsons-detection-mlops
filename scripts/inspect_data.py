
import pandas as pd
from pathlib import Path


def main():

    file_path = Path("dataset/raw/pd_speech_features.csv")

    print("Loading Parkinson's dataset...")

    # Read the CSV using the correct header row
    df = pd.read_csv(file_path, header=1)

    print("\n========== DATASET OVERVIEW ==========")

    print("Dataset shape:", df.shape)

    print("\nFirst 15 columns:")
    print(df.columns.tolist()[:15])

    print("\nFirst 5 records:")
    print(df.head())

    print("\nDataset information:")
    df.info()

    print("\nMissing values:", df.isnull().sum().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nParticipant information:")
    print("Unique participants:", df["id"].nunique())

    print("\nGender distribution:")
    print(df["gender"].value_counts())

    print("\nClass distribution:")
    print(df["class"].value_counts())

    print("\nClass percentages:")
    print(df["class"].value_counts(normalize=True) * 100)

    print("\nStatistical summary:")
    print(df.describe())

    print("\nInspection completed!")


if __name__ == "__main__":
    main()
