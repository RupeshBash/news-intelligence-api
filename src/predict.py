# src/predict.py

# joblib is used to load the saved model package
import joblib

# pathlib helps us create safe file paths
from pathlib import Path


# Get project root folder
# __file__ means current file path: src/predict.py
# parents[1] goes one level up from src/ to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Full path of saved model file
MODEL_PATH = PROJECT_ROOT / "models" / "news_classifier_pipeline.joblib"


def clean_text(text):
    """
    Clean input text before prediction.

    This should be similar to the cleaning used during training.
    """
    # Replace backslash with space
    text = text.replace("\\", " ")

    # Replace new line with space
    text = text.replace("\n", " ")

    # Remove extra spaces from start and end
    text = text.strip()

    # Return cleaned text
    return text


def load_model_package():
    """
    Load saved vectorizer, model, and label mapping.
    """
    # Check if model file exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Load saved package from joblib file
    model_package = joblib.load(MODEL_PATH)

    return model_package


def predict_category(text):
    """
    Predict news category for one input text.
    """
    # Load saved package
    model_package = load_model_package()

    # Get vectorizer from saved package
    vectorizer = model_package["vectorizer"]

    # Get trained model from saved package
    model = model_package["model"]

    # Get class ID to label name mapping
    id_to_label = model_package["id_to_label"]

    # Clean user input text
    cleaned_text = clean_text(text)

    # Convert text into TF-IDF numeric features
    # Vectorizer expects a list, so we use [cleaned_text]
    text_tfidf = vectorizer.transform([cleaned_text])

    # Predict class ID
    prediction = model.predict(text_tfidf)

    # prediction is an array, so take first item
    predicted_id = prediction[0]

    # Convert predicted ID into readable label
    predicted_label = id_to_label[predicted_id]

    return predicted_id, predicted_label


# This part runs only when we directly run:
# python src/predict.py
if __name__ == "__main__":
    # Sample news text for testing
    sample_text = "Apple announces new AI features for iPhone users"

    # Predict category
    predicted_id, predicted_label = predict_category(sample_text)

    # Print result
    print("Input text:", sample_text)
    print("Predicted class ID:", predicted_id)
    print("Predicted category:", predicted_label)