"""
Reusable training/evaluation pipeline for the Personal Loan Prediction project.

The raw dataset is intentionally not committed to the repository.
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


TARGET = "Personal Loan"
DROP_FOR_MODEL = ["Personal Loan", "ZIP Code", "Latitude", "Longitude"]


def load_and_prepare(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    # Remove identifier/redundant feature used in the original notebook.
    df = df.drop(columns=["ID", "Experience"], errors="ignore")

    # Original dataset stores decimal values with "/" in some CCAvg entries.
    if df["CCAvg"].dtype == object:
        df["CCAvg"] = df["CCAvg"].astype(str).str.replace("/", ".", regex=False).astype(float)

    # Convert monthly credit-card spending to yearly spending, matching the notebook.
    df["CCAvg"] = df["CCAvg"] * 12
    return df


def split_features(df: pd.DataFrame):
    X = df.drop(columns=DROP_FOR_MODEL, errors="ignore")
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.30, random_state=0)


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def run_models(df: pd.DataFrame) -> dict:
    X_train, X_test, y_train, y_test = split_features(df)

    results = {}

    # Logistic Regression — mirrors the notebook's baseline.
    logreg = LogisticRegression(solver="liblinear", max_iter=2000)
    logreg.fit(X_train, y_train)
    results["logistic_regression"] = evaluate_predictions(
        y_test, logreg.predict(X_test)
    )

    # Gaussian Naive Bayes baseline.
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    results["gaussian_nb"] = evaluate_predictions(
        y_test, gnb.predict(X_test)
    )

    # KNN is distance-based, so scaling is applied before fitting.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_s, y_train)
    results["knn_scaled_k3"] = evaluate_predictions(
        y_test, knn.predict(X_test_s)
    )

    cv_scores = cross_val_score(knn, X_train_s, y_train, cv=5)
    results["knn_scaled_k3"]["cv_mean_accuracy"] = float(cv_scores.mean())

    return results


if __name__ == "__main__":
    data_path = Path("data/Bank_Personal_Loan_Modelling.csv")
    dataframe = load_and_prepare(data_path)
    model_results = run_models(dataframe)

    for name, metrics in model_results.items():
        print(f"\n{name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        if "cv_mean_accuracy" in metrics:
            print(f"CV mean accuracy: {metrics['cv_mean_accuracy']:.4f}")
