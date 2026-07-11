# AI News Classifier + Similar Article Search API

This project classifies news articles into categories and retrieves semantically similar articles.

## Day 1 Progress

- Loaded Kaggle AG News dataset
- Checked dataset shape, columns, labels, and missing values
- Renamed columns for cleaner Python usage
- Mapped class numbers to readable category names
- Combined title and description into one text field
- Performed light text cleanup
- Calculated character count and word count
- Saved a small sample CSV for future testing

## Day 2 Progress

- Created a balanced subset of the AG News dataset
- Used 2,000 rows from each class
- Split data into train and test sets
- Converted text into TF-IDF features
- Trained a Logistic Regression baseline model
- Achieved 88.9% test accuracy
- Tested prediction on custom news text

## Day 3 Progress

- Evaluated the baseline model using accuracy, precision, recall, and F1-score
- Created a confusion matrix to inspect classification mistakes
- Saved the trained vectorizer, model, and label mapping using joblib
- Loaded the saved model package and tested prediction again
- Created `src/predict.py` for reusable prediction
- Achieved 89.25% test accuracy

## Day 4 Progress

- Compared Logistic Regression with Multinomial Naive Bayes
- Used the same balanced dataset and train/test split for fair comparison
- Evaluated both models using accuracy, precision, recall, F1-score, and confusion matrix
- Compared predictions on custom news text
- Naive Bayes performed bettern than Logistic Regression.