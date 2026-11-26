
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier


PROCESSED_DIR = Path("dataset/processed")
OUTPUT_DIR = Path("reports/model_comparison")

RANDOM_STATE = 42
TOP_FEATURES = 100
N_SPLITS = 5


def load_data():

    print("Loading original training data...")

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).squeeze("columns")

    groups_train = pd.read_csv(
        PROCESSED_DIR / "groups_train.csv"
    ).squeeze("columns")

    print("Training data shape:", X_train.shape)
    print("Training labels:", y_train.shape)
    print("Unique participants:", groups_train.nunique())

    return X_train, y_train, groups_train


def create_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=False,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        )
    }

    return models


def create_pipeline(model):

    feature_selector = SelectFromModel(
        estimator=RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        max_features=TOP_FEATURES,
        threshold=-float("inf")
    )

    pipeline = Pipeline([
        ("feature_selection", feature_selector),

        ("scaling", StandardScaler()),

        ("smote", SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=5
        )),

        ("classifier", model)
    ])

    return pipeline


def evaluate_models(X_train, y_train, groups_train):

    print("\nPreparing participant-aware cross-validation...")

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    models = create_models()

    results = []

    for name, model in models.items():

        print("\n" + "=" * 55)
        print("Training:", name)
        print("=" * 55)

        pipeline = create_pipeline(model)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise"
        )

        result = {
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1 Score": scores["test_f1"].mean(),
            "ROC AUC": scores["test_roc_auc"].mean()
        }

        results.append(result)

        print(f"Accuracy:  {result['Accuracy']:.4f}")
        print(f"Precision: {result['Precision']:.4f}")
        print(f"Recall:    {result['Recall']:.4f}")
        print(f"F1 Score:  {result['F1 Score']:.4f}")
        print(f"ROC AUC:   {result['ROC AUC']:.4f}")

    return pd.DataFrame(results)


def save_results(results):

    print("\nSaving model comparison results...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = results.sort_values(
        by="ROC AUC",
        ascending=False
    )

    results.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    print("\n========== MODEL COMPARISON ==========")

    print(
        results.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format,
                "ROC AUC": "{:.4f}".format
            }
        )
    )

    print("\nResults saved to:", OUTPUT_DIR)


def main():

    X_train, y_train, groups_train = load_data()

    results = evaluate_models(
        X_train,
        y_train,
        groups_train
    )

    save_results(results)

    print("\n========== MODEL TRAINING COMPLETE ==========")


if __name__ == "__main__":
    main()
