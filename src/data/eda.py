
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


def main():

    print("Starting Exploratory Data Analysis...")

    # 1. Load dataset
    df = pd.read_csv(
        "dataset/raw/pd_speech_features.csv",
        header=1
    )

    output_dir = Path("reports/eda")
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Identify speech features
    feature_columns = [
        col for col in df.columns
        if col not in ["id", "gender", "class"]
    ]

    X = df[feature_columns]

    # 2. Dataset overview
    print("\n========== DATASET OVERVIEW ==========")

    print("Dataset shape:", df.shape)
    print("Total recordings:", len(df))
    print("Unique participants:", df["id"].nunique())
    print("Speech features:", len(feature_columns))
    print("Missing values:", df.isnull().sum().sum())

    # 3. Class distribution
    print("\n========== CLASS DISTRIBUTION ==========")

    print(df["class"].value_counts())

    print("\nClass percentages:")
    print(df["class"].value_counts(normalize=True) * 100)

    plt.figure(figsize=(7, 5))

    sns.countplot(data=df, x="class")

    plt.title("Parkinson's Disease Class Distribution")
    plt.xlabel("Class (0 = Healthy, 1 = Parkinson's)")
    plt.ylabel("Number of Recordings")

    plt.savefig(
        output_dir / "class_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # 4. Participant analysis
    print("\n========== PARTICIPANT ANALYSIS ==========")

    participant_counts = df.groupby("id").size()

    print("Unique participants:", df["id"].nunique())

    print("\nRecordings per participant:")
    print(participant_counts.value_counts().sort_index())

    participant_labels = df.groupby("id")["class"].nunique()

    print(
        "\nParticipants with inconsistent class labels:",
        (participant_labels > 1).sum()
    )

    # 5. Duplicate analysis
    print("\n========== DUPLICATE ANALYSIS ==========")

    print("Exact duplicate rows:", df.duplicated().sum())

    duplicate_features = X.duplicated(keep=False)

    print(
        "Records sharing identical speech features:",
        duplicate_features.sum()
    )

    # 6. Duplicate investigation
    print("\n========== DUPLICATE INVESTIGATION ==========")

    exact_duplicates = df[
        df.duplicated(keep=False)
    ]

    print("\nExact duplicate records:")

    if exact_duplicates.empty:
        print("No exact duplicate records found.")
    else:
        print(
            exact_duplicates[
                ["id", "gender", "class"]
            ].to_string(index=False)
        )

    feature_duplicates = df[
        df.duplicated(
            subset=feature_columns,
            keep=False
        )
    ]

    print("\nRecords with identical speech features:")

    if feature_duplicates.empty:
        print("No identical speech feature records found.")
    else:
        print(
            feature_duplicates[
                ["id", "gender", "class"]
            ].to_string(index=False)
        )

    # 7. Feature statistics
    print("\n========== FEATURE STATISTICS ==========")

    print("Number of speech features:", X.shape[1])

    print("\nFeature summary:")
    print(X.describe().T.head(15))

    # 8. Feature distributions
    print("\n========== FEATURE DISTRIBUTIONS ==========")

    selected_features = X.var().nlargest(6).index

    for feature in selected_features:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=feature,
            hue="class",
            kde=True,
            element="step"
        )

        plt.title(f"Distribution of {feature}")
        plt.tight_layout()

        plt.savefig(
            output_dir / f"{feature}_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # 9. Correlation analysis
    print("\n========== CORRELATION ANALYSIS ==========")

    top_features = X.var().nlargest(15).index

    correlation = df[top_features].corr()

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=False
    )

    plt.title("Speech Feature Correlation Heatmap")

    plt.savefig(
        output_dir / "correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # 10. Outlier analysis
    print("\n========== OUTLIER ANALYSIS ==========")

    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)

    IQR = Q3 - Q1

    outlier_counts = (
        (X < Q1 - 1.5 * IQR) |
        (X > Q3 + 1.5 * IQR)
    ).sum()

    print(
        "Features containing outliers:",
        (outlier_counts > 0).sum()
    )

    print("\nTop 10 features by outlier count:")
    print(outlier_counts.nlargest(10))

    # 11. Save feature analysis report
    print("\n========== SAVING REPORTS ==========")

    summary = pd.DataFrame({
        "Feature": X.columns,
        "Mean": X.mean().values,
        "Standard Deviation": X.std().values,
        "Outlier Count": outlier_counts.values
    })

    summary.to_csv(
        output_dir / "feature_analysis.csv",
        index=False
    )

    print("\nEDA completed successfully!")

    print("Charts and reports saved inside:", output_dir)


if __name__ == "__main__":
    main()