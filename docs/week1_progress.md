# Week 1 Development Progress

This document contains the day-by-day development record for the classical
machine-learning classification stage of the AI News Classifier project.

## Day 1 Progress — Dataset Exploration

- Loaded the Kaggle AG News dataset
- Checked dataset shape, columns, labels, and missing values
- Renamed columns for cleaner Python usage
- Mapped numeric class IDs to readable category names
- Combined title and description into one text field
- Performed light text cleaning
- Calculated character count and word count
- Saved a small sample CSV for future testing

### Important decisions

Original columns:

```text
Class Index
Title
Description
```

Renamed columns:

```text
class_index
title
description
```

Label mapping:

```text
1 → World
2 → Sports
3 → Business
4 → Sci/Tech
```

Combined text:

```python
text = title + " " + description
```

Light cleaning included:

- replacing backslashes with spaces
- replacing newline characters with spaces
- removing leading and trailing spaces

### Main insight

Heavy preprocessing was not added before creating the baseline model. The goal
was to start with a simple and understandable pipeline.

---

## Day 2 Progress — Baseline Model

- Created a balanced subset of the AG News dataset
- Used 2,000 rows from each category
- Created 8,000 total rows
- Split the data into training and test sets
- Converted text into TF-IDF features
- Trained a Logistic Regression baseline
- Achieved approximately 88.9% test accuracy
- Tested prediction using custom news text

### Baseline configuration

```python
TfidfVectorizer(
    max_features=10000,
    lowercase=True,
    stop_words=None,
)

LogisticRegression(max_iter=1000)
```

### Main workflow

```text
Raw text
    ↓
TF-IDF numeric features
    ↓
Logistic Regression
    ↓
Predicted news category
```

### Main insight

Machine-learning models cannot directly use raw text. TF-IDF converts text into
numeric word-importance features that Logistic Regression can use.

---

## Day 3 Progress — Evaluation and Model Saving

- Evaluated the model using accuracy, precision, recall, and F1-score
- Created a confusion matrix
- Inspected common model mistakes
- Saved the trained vectorizer, model, and label mapping using joblib
- Loaded the saved package and tested prediction again
- Created `src/predict.py` for reusable prediction
- Achieved 89.25% test accuracy in one evaluation run

### Evaluation result

```text
World       precision 0.92  recall 0.89  f1-score 0.90
Sports      precision 0.91  recall 0.97  f1-score 0.94
Business    precision 0.88  recall 0.86  f1-score 0.87
Sci/Tech    precision 0.87  recall 0.85  f1-score 0.86
```

Main confusion:

```text
Business ↔ Sci/Tech
```

### Saved model package

```python
model_package = {
    "vectorizer": vectorizer,
    "model": model,
    "id_to_label": id_to_label,
}
```

### Main insight

The trained model is not enough by itself. Prediction must use the same
vectorizer because it contains the vocabulary and feature positions learned
during training.

---

## Day 4 Progress — Model Comparison

- Compared Logistic Regression with Multinomial Naive Bayes
- Used the same balanced dataset
- Used the same train/test split
- Used the same TF-IDF features
- Printed classification reports for both models
- Compared their confusion patterns
- Tested both models using custom news text

### Comparison result

| Model | Accuracy |
|---|---:|
| Logistic Regression | 85.69% |
| Multinomial Naive Bayes | 85.81% |

Multinomial Naive Bayes performed slightly better, but the difference was only
about 0.12 percentage points.

### Main insight

A very small accuracy difference should not be overclaimed. Both models
performed almost the same.

A fair model comparison requires:

```text
same dataset
same train/test split
same vectorizer
same evaluation metrics
```

---

## Day 5 Progress — Reusable Python Scripts

- Refactored notebook code into reusable Python scripts
- Created `src/preprocessing.py`
- Created `src/train_model.py`
- Updated `src/predict.py`
- Saved the TF-IDF vectorizer, classifier, and label mapping together
- Confirmed that prediction works without retraining
- Learned to run project modules from the project root

### Main workflow

```text
preprocessing.py → clean and prepare text
train_model.py   → train and save the model
predict.py       → load the saved model and predict
```

### Main files

#### `src/preprocessing.py`

Contains reusable text-preparation functions such as:

```text
clean_text()
combine_title_description()
prepare_text_column()
```

#### `src/train_model.py`

Responsible for:

- loading the dataset
- preparing balanced data
- splitting training and test data
- creating TF-IDF features
- training the classifier
- evaluating the model
- saving the model package

#### `src/predict.py`

Responsible for:

- loading the saved model package
- cleaning new text
- transforming it using the saved vectorizer
- predicting the category
- returning a readable label

### Commands

```powershell
python -m src.preprocessing
python -m src.train_model
python -m src.predict
```

### Main insight

```text
Train once
Save the trained components
Load them later for prediction
```

---

## Day 6 Progress — Streamlit Interface

- Created a Streamlit interface for news-category prediction
- Added a text area for news input
- Added a Predict Category button
- Connected the interface to `src/predict.py`
- Loaded the saved joblib model
- Added prediction confidence using `predict_proba()`
- Displayed the predicted category
- Displayed confidence percentage
- Displayed the predicted class ID
- Added empty-input validation
- Added basic prediction error handling
- Confirmed that Streamlit does not retrain the model

### Day 6 workflow

```text
User enters news text
        ↓
User clicks Predict Category
        ↓
Streamlit validates the input
        ↓
Text is sent to src/predict.py
        ↓
Saved TF-IDF vectorizer transforms the text
        ↓
Saved Logistic Regression model predicts the category
        ↓
predict_proba() provides probabilities
        ↓
Streamlit displays category and confidence
```

### Main responsibilities

```text
src/predict.py
    → prediction logic

app/streamlit_app.py
    → browser interface

models/news_classifier_pipeline.joblib
    → saved trained components
```

### Streamlit command

```powershell
python -m streamlit run app\streamlit_app.py
```

### Validation

If the user enters no text, the application shows:

```text
Please enter a news headline or article.
```

### Example test

Input:

```text
The device can process complex tasks faster while using less energy than the
previous version.
```

Result:

```text
Predicted category: Sci/Tech
Confidence: 81.83%
```

### Main insight

Streamlit is the user-interface layer. The saved machine-learning model performs
the prediction.

Training and inference are separate:

```text
train_model.py     → training
streamlit_app.py   → inference through a user interface
```

---

## Day 7 Progress — Documentation and Week 1 Review

- Reviewed the complete Week 1 machine-learning pipeline
- Cleaned and reorganized the main README
- Separated completed features from planned features
- Documented the model architecture
- Documented file responsibilities
- Added training configuration
- Added evaluation results
- Added the model-comparison result
- Added Streamlit screenshots
- Added the confusion-matrix screenshot
- Added installation and run instructions
- Added current limitations
- Added future improvements
- Moved detailed daily progress into this document
- Practised explaining the complete Week 1 project

### Week 1 complete workflow

```text
AG News dataset
        ↓
Dataset exploration
        ↓
Column cleaning and label mapping
        ↓
Title and description combined
        ↓
Balanced subset
        ↓
Train/test split
        ↓
TF-IDF vectorization
        ↓
Logistic Regression training
        ↓
Evaluation and confusion matrix
        ↓
Model saved using joblib
        ↓
Reusable prediction scripts
        ↓
Streamlit interface
```

### Week 1 short explanation

During Week 1, I built the classification part of my AI News Intelligence
project using the AG News dataset. I cleaned and prepared the data, converted
the articles into TF-IDF features, and trained a Logistic Regression model.

I evaluated the classifier using accuracy, precision, recall, F1-score, and a
confusion matrix. I also compared Logistic Regression with Multinomial Naive
Bayes.

I saved the vectorizer, classifier, and label mapping using joblib. Then I
refactored the notebook code into reusable Python scripts and connected the
saved classifier to a Streamlit application.

The Streamlit app accepts news text and displays the predicted category and
confidence without retraining the model.

### Week 1 main takeaway

The classification stage is now complete and reusable.

```text
Dataset preparation
        ↓
Feature extraction
        ↓
Model training
        ↓
Model evaluation
        ↓
Model saving
        ↓
Prediction script
        ↓
Streamlit application
```

The next project stage will add semantic embeddings and similar-article search.