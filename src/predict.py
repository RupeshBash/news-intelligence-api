from pathlib import Path

import joblib

from src.preprocessing import clean_text



#Project root: news-intelligence-api
PROJECT_ROOT = Path(__file__).resolve().parents[1]


#saved root: news-intelligence-api/
MODEL_PATH = PROJECT_ROOT / "models" / "news_classifier_pipeline.joblib"


def load_model_package():
    """Load saved vectorizer, model, and label mapping."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)

def predict_category(text):
    """Predict news category for one input text."""
    model_package = load_model_package()

    vectorizer = model_package["vectorizer"]
    model = model_package["model"]
    id_to_label = model_package["id_to_label"]

    cleaned_text = clean_text(text)

    #convert single text into TF-IDF features
    text_tfidf = vectorizer.transform([cleaned_text]) #vectorizer.transform([cleaned_text]) uses the same TF_IDF vocabulary learned during training.

    #predict class ID and convert it to category name
    predicted_id = model.predict(text_tfidf)[0]
    predicted_label = id_to_label[predicted_id]

    return predicted_id, predicted_label

if __name__ == "__main__":
    sample_text = "Apple announces new AI features for iphone users"

    predicted_id, predicted_label = predict_category(sample_text)

    print("Input text:", sample_text)
    print("Predicted class ID:", predicted_id)
    print("Predicted category:", predicted_label)