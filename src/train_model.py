from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.preprocessing import prepare_text_column


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "news_classifier_pipeline.joblib"

ROWS_PER_CLASS = 6000
MAX_FEATURES = 10000

LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}


def load_data():
    """Load and rename AG News data."""
    df = pd.read_csv(DATA_PATH)

    df = df.rename(columns={
        "Class Index": "class_index",
        "Title": "title",
        "Description": "description",
    })

    df["label_name"] = df["class_index"].map(LABEL_MAP)

    return df


def create_balanced_subset(df, rows_per_class):
    """Create equal number of rows for each class."""
    balanced_df = (
        df.groupby("class_index", group_keys=False)
        .sample(n=rows_per_class, random_state=42)
    )

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return balanced_df


def train_model():
    """Train, evaluate, and save the classifier."""
    df = load_data()
    df = prepare_text_column(df)

    balanced_df = create_balanced_subset(
        df,
        rows_per_class=ROWS_PER_CLASS
    )

    print("Rows per class:", ROWS_PER_CLASS)
    print("Total dataset size:", len(balanced_df))
    print("Class distribution:")
    print(balanced_df["label_name"].value_counts())

    X = balanced_df["text"]
    y = balanced_df["class_index"]

    # Keep class balance in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        lowercase=True,
        stop_words=None,
    )

    # Fit only on training text to avoid data leakage
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Train TF-IDF shape:", X_train_tfidf.shape)
    print("Test TF-IDF shape:", X_test_tfidf.shape)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    print("Classification report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[1, 2, 3, 4],
        target_names=["World", "Sports", "Business", "Sci/Tech"],
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test,
        y_pred,
        labels=[1, 2, 3, 4],
    ))

    # Save vectorizer with model because prediction needs the same TF-IDF vocabulary
    model_package = {
        "vectorizer": vectorizer,
        "model": model,
        "id_to_label": LABEL_MAP,
        "rows_per_class": ROWS_PER_CLASS,
        "max_features": MAX_FEATURES,
        "accuracy": accuracy,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_package, MODEL_PATH)

    print("Model saved to:")
    print(MODEL_PATH)


if __name__ == "__main__":
    train_model()