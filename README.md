# AI News Classifier + Similar Article Search API

A laptop-friendly AI News Intelligence project that classifies English news
text and retrieves semantically similar stored articles.

The classifier uses TF-IDF and Logistic Regression to predict one of four AG
News categories. The semantic-search workflow uses MiniLM embeddings and a
persistent ChromaDB collection containing 1,000 sampled articles.

The combined workflow is available through a Streamlit web interface.

A FastAPI application is also available with shared resource loading,
a `GET /health` endpoint, and a validated `POST /predict` endpoint.

---

## Current Features

- Accepts a news headline or short article
- Performs light text cleaning
- Predicts one of four AG News categories
- Displays the predicted class ID
- Displays prediction confidence
- Uses a saved TF-IDF vectorizer and Logistic Regression model
- Generates 384-dimensional MiniLM query embeddings
- Searches a persistent ChromaDB collection
- Retrieves the top five semantically similar articles
- Displays article category, cosine distance, and cosine similarity
- Combines classification and retrieval through `analyze_news()`
- Caches MiniLM and ChromaDB resources in Streamlit
- Handles empty input, missing values, and processing errors
- Loads saved resources without retraining during prediction
- Provides a FastAPI application
- Loads MiniLM and ChromaDB once during FastAPI startup
- Provides a `GET /health` endpoint
- Provides a validated `POST /predict` endpoint
- Returns class ID, category, and prediction confidence through FastAPI
- Uses Pydantic models for prediction request and response validation
- Provides interactive FastAPI documentation through `/docs`

---

## Supported Categories

| Class ID | Category |
|---:|---|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

---

## System Workflow

```text
User enters news text
        |
        +--> TF-IDF vectorizer
        |        |
        |        +--> Logistic Regression
        |                 |
        |                 +--> category + confidence
        |
        +--> MiniLM embedding model
                 |
                 +--> 384-dimensional query embedding
                          |
                          +--> ChromaDB cosine search
                                   |
                                   +--> top five similar articles
        |
        +--> Combined result displayed in Streamlit
```

The reusable prediction and search functions are also being exposed through
FastAPI endpoints during Week 3.

Current API flow:

```text
Client request
      ↓
FastAPI
      ↓
Pydantic validation
      ↓
existing ML function
      ↓
structured JSON response
```

---

## Main Files

| File | Responsibility |
|---|---|
| `src/preprocessing.py` | Cleans and prepares news text |
| `src/train_model.py` | Trains, evaluates, and saves the classifier |
| `src/predict.py` | Loads the saved model and predicts category and confidence |
| `src/embedding_demo.py` | Demonstrates MiniLM embeddings and cosine similarity |
| `src/generate_embeddings.py` | Generates embeddings for sampled AG News articles |
| `src/store_embeddings.py` | Stores article embeddings and metadata in ChromaDB |
| `src/search_articles.py` | Embeds a query and retrieves similar articles |
| `src/analyze_news.py` | Combines classification and semantic retrieval |
| `app/streamlit_app.py` | Displays the combined workflow through Streamlit |
| `app/fastapi_app.py` | Creates the FastAPI application, shared resources, validation models, health endpoint, and prediction endpoint |
| `models/news_classifier_pipeline.joblib` | Stores the vectorizer, classifier, and label mapping |
| `docs/week1_progress.md` | Tracks Week 1 classical ML development |
| `docs/week2_progress.md` | Tracks Week 2 embedding and semantic-search development |
| `docs/week3_progress.md` | Tracks Week 3 FastAPI and API development |

---

## Training Configuration

The stronger saved classifier currently uses:

- Dataset: AG News
- Rows per category: 6,000
- Total rows: 24,000
- Training rows: 19,200
- Test rows: 4,800
- Train/test ratio: 80/20
- TF-IDF maximum features: 10,000
- Classifier: Logistic Regression
- Maximum iterations: 1,000

---

## Model Results

### Baseline Evaluation

One baseline evaluation produced:

- Test accuracy: 89.25%
- World F1-score: 0.90
- Sports F1-score: 0.94
- Business F1-score: 0.87
- Sci/Tech F1-score: 0.86

The main confusion occurred between Business and Sci/Tech because these
categories can share vocabulary related to companies, products, software,
markets, and technology.

### Model Comparison

Logistic Regression and Multinomial Naive Bayes were compared using the same
prepared dataset, train/test split, and TF-IDF features.

| Model | Accuracy |
|---|---:|
| Logistic Regression | 85.69% |
| Multinomial Naive Bayes | 85.81% |

Multinomial Naive Bayes performed slightly better in this experiment, but the
difference was very small. Both models performed almost the same.

Results varied slightly between separately prepared experiment runs. A fair
comparison requires the same dataset split and feature representation.

---

## Semantic Search Configuration

The current semantic-search stage uses:

- Embedding model: `all-MiniLM-L6-v2`
- Embedding dimension: 384
- Indexed articles: 1,000
- Articles per category: 250
- ChromaDB collection: `ag_news_articles`
- Persistent database path: `data/chroma_db`
- Distance method: Cosine
- Default results returned: 5

The query text is converted into a MiniLM embedding and compared with stored
article embeddings.

For the current cosine configuration:

```text
Lower cosine distance
→ closer semantic match

Higher cosine similarity
→ closer semantic match
```

Cosine similarity is not the same as classifier confidence.

---

## Classification and Retrieval

The project combines two related but different tasks.

### Classification

Classification answers:

```text
Which AG News category best describes this input?
```

It returns:

- Class index
- Category label
- Confidence percentage

### Semantic Retrieval

Semantic retrieval answers:

```text
Which stored articles are closest to this input in embedding space?
```

It returns:

- Rank
- Article title
- Article category
- Article ID
- Cosine distance
- Cosine similarity

The predicted category and the category of the nearest stored article do not
always need to match because classification and retrieval solve different tasks.

---

## Project Structure

```text
news-intelligence-api/
├── app/
│   ├── streamlit_app.py
│   └── fastapi_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── article_embeddings.npy
│   │   └── article_metadata.csv
│   └── chroma_db/
├── docs/
│   ├── week1_progress.md
│   ├── week2_progress.md
│   └── week3_progress.md
├── models/
│   └── news_classifier_pipeline.joblib
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiment.ipynb
│   ├── 03_evaluation_model_saving.ipynb
│   └── 04_model_comparison.ipynb
├── screenshots/
│   ├── confusion_matrix.png
│   ├── streamlit_home.png
│   ├── streamlit_prediction.png
│   ├── streamlit_validation.png
│   └── day13_combined_streamlit_ui.png
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   ├── embedding_demo.py
│   ├── generate_embeddings.py
│   ├── store_embeddings.py
│   ├── search_articles.py
│   └── analyze_news.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

Generated embeddings, metadata, raw data, local ChromaDB files, Python caches,
and virtual-environment files remain ignored by Git.

---

## Technology Stack

- Python
- Pandas
- NumPy
- Hugging Face Datasets
- scikit-learn
- joblib
- SentenceTransformers
- ChromaDB
- Streamlit
- FastAPI
- Pydantic
- Uvicorn
- Matplotlib
- Jupyter Notebook
- Git and GitHub

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/RupeshBash/news-intelligence-api.git
cd news-intelligence-api
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment on Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Prepare Local Project Resources

Generated resources are ignored by Git and may need to be created locally after
cloning the repository.

### Train and Save the Classifier

```powershell
python -m src.train_model
```

This creates the local saved model package used during prediction.

### Generate Article Embeddings

```powershell
python -m src.generate_embeddings
```

This creates:

```text
data/processed/article_embeddings.npy
data/processed/article_metadata.csv
```

### Store Articles in ChromaDB

```powershell
python -m src.store_embeddings
```

This creates the persistent local ChromaDB collection.

---

## Run the Project

Run all commands from the project root.

### Run Standalone Classification

```powershell
python -m src.predict
```

### Run the Embedding Demonstration

```powershell
python -m src.embedding_demo
```

### Run Standalone Semantic Search

```powershell
python -m src.search_articles
```

### Run Combined Classification and Retrieval

```powershell
python -m src.analyze_news
```

### Run the Streamlit Interface

```powershell
python -m streamlit run app\streamlit_app.py
```

The Streamlit application normally opens at:

```text
http://localhost:8501
```

### Run the FastAPI Application

```powershell
python -m uvicorn app.fastapi_app:app --reload
```

The API normally runs at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Current endpoints:

```text
GET  /health
POST /predict
```

---

## Sample Combined Analysis

Input:

```text
A company introduced a new computer chip that uses less electricity.
```

Example classification output:

```text
Predicted class index: 4
Predicted category: Sci/Tech
Prediction confidence: 89.94%
```

Example semantic-search output:

```text
Similar articles returned: 5
Embedding shape: (384,)
```

The exact confidence, search time, and retrieved articles depend on the saved
model and current ChromaDB collection.

---

## Streamlit Interface

The Streamlit application accepts a news headline or short article and
displays:

- Predicted class index
- Predicted category
- Confidence percentage
- Vector-search time
- Five semantically similar articles
- Article category
- Cosine distance
- Cosine similarity
- Article ID

The MiniLM model and ChromaDB resources are cached using `st.cache_resource`,
preventing unnecessary reloading during Streamlit reruns.

![AI News Intelligence Streamlit interface](screenshots/day13_combined_streamlit_ui.png)

---

## FastAPI Interface

The FastAPI application exposes reusable project functionality through HTTP
endpoints.

The application uses a startup lifespan to load shared resources once:

- MiniLM embedding model
- ChromaDB client
- ChromaDB article collection
- Indexed article count

This avoids loading expensive resources again for every HTTP request.

### Current Endpoints

```text
GET  /health
POST /predict
```

---

### GET /health

The health endpoint confirms that the API and shared resources are ready.

It reports information such as:

- API status
- Service name
- Embedding-model readiness
- ChromaDB collection name
- Indexed article count

Example:

```text
GET /health
```

---

### POST /predict

The prediction endpoint accepts news text and returns the classifier result.

Example request:

```json
{
  "news_text": "A company introduced a new computer chip that uses less electricity."
}
```

Example response:

```json
{
  "input_text": "A company introduced a new computer chip that uses less electricity.",
  "prediction": {
    "class_index": 4,
    "category": "Sci/Tech",
    "confidence_percent": 89.94
  }
}
```

The exact confidence depends on the saved classifier.

The endpoint:

```text
receives JSON
      ↓
validates news_text
      ↓
cleans unnecessary whitespace
      ↓
calls predict_category_with_confidence()
      ↓
returns structured JSON
```

The FastAPI layer reuses the existing classifier instead of duplicating the
machine-learning logic.

---

## API Validation

The prediction endpoint uses Pydantic models to define request and response
contracts.

The request requires:

```text
news_text
→ string
→ minimum useful content required
→ maximum length: 5,000 characters
```

Invalid requests such as:

```text
empty news_text
whitespace-only news_text
missing news_text
incorrect data type
```

are rejected before the prediction function is called.

---

## Interactive API Documentation

FastAPI automatically provides interactive documentation at:

```text
http://127.0.0.1:8000/docs
```

The documentation currently exposes:

```text
GET  /health
POST /predict
```

It also displays the request and response schemas generated from the Pydantic
models.

---

## Planned API Endpoints

```text
POST /similar
POST /analyze
```

These endpoints will reuse the existing semantic-search and combined-analysis
functions rather than duplicating processing logic.

---

## Additional Screenshots

### Classifier Home Page

![Streamlit home page](screenshots/streamlit_home.png)

### Classification Result

![Streamlit prediction result](screenshots/streamlit_prediction.png)

### Confusion Matrix

![Confusion matrix](screenshots/confusion_matrix.png)

---

## Current Limitations

- The classifier supports only four AG News categories.
- The classifier was trained mainly on English news text.
- The semantic index contains only 1,000 sampled articles.
- Very short or ambiguous input may produce unreliable results.
- High classifier confidence does not guarantee correctness.
- High cosine similarity does not guarantee genuine relevance.
- TF-IDF does not deeply understand semantic meaning.
- The classifier and nearest article may have different category labels.
- Some development messages are still printed in the terminal.
- The application currently runs locally.
- FastAPI semantic-search and combined-analysis endpoints are still pending.
- Automated API and integration tests have not yet been added.

---

## Planned Improvements

- Add `POST /similar`
- Add `POST /analyze`
- Add request and response validation for the remaining endpoints
- Add automated API and integration tests
- Add a basic Dockerfile
- Improve semantic-search evaluation
- Increase the number of indexed articles when hardware permits

---

## Development Progress

- [View Week 1 Progress](docs/week1_progress.md)
- [View Week 2 Progress](docs/week2_progress.md)
- [View Week 3 Progress](docs/week3_progress.md)

---

## Current Project Status

```text
Week 1
Classical ML news classifier
Status: Complete

Week 2
Embeddings, ChromaDB, semantic search, combined analysis, and Streamlit
Status: Complete

Week 3
FastAPI, testing, Docker basics, and final project polish
Status: In Progress

Day 15
FastAPI application foundation, shared startup resources, and GET /health
Status: Complete

Day 16
POST /predict with Pydantic validation and structured prediction response
Status: Complete
```