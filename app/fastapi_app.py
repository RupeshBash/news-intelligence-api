# -------------------- IMPORTS

from contextlib import asynccontextmanager

from fastapi import FastAPI

from pydantic import (
    BaseModel,
    Field,
    StrictInt,
    field_validator,
)

from src.predict import (
    predict_category_with_confidence,
)

from src.search_articles import (
    load_embedding_model,
    load_chroma_collection,
    prepare_query_text,
    generate_query_embedding,
    search_chroma_Collection,
    format_search_results,
)


# -------------------- REQUEST MODELS

class PredictRequest(BaseModel):
    """Expected JSON body for prediction."""

    # Client must send at least one character.
    news_text: str = Field(
        min_length=1,
        max_length=5000,
    )

    @field_validator("news_text")
    @classmethod
    def clean_news_text(cls, value):
        """Clean and reject empty text."""

        # Remove repeated spaces and line breaks.
        cleaned_text = " ".join(
            value.split()
        )

        # Reject whitespace-only input.
        if not cleaned_text:
            raise ValueError(
                "news_text cannot be empty."
            )

        return cleaned_text


class SimilarRequest(BaseModel):
    """Input required for semantic search."""

    # News text sent by the client.
    news_text: str = Field(
        min_length=1,
        max_length=5000,
    )

    # Number of similar articles to request.
    top_k: StrictInt = Field(
        default=5,
        ge=1,
        le=10,
    )

    @field_validator("news_text")
    @classmethod
    def clean_news_text(cls, value):
        """Clean and reject empty text."""

        # Remove repeated spaces and line breaks.
        cleaned_text = " ".join(
            value.split()
        )

        # Reject whitespace-only input.
        if not cleaned_text:
            raise ValueError(
                "news_text cannot be empty."
            )

        return cleaned_text


# -------------------- PREDICTION RESPONSE MODELS

class PredictionResult(BaseModel):
    """Prediction details."""

    class_index: int
    category: str
    confidence_percent: float


class PredictResponse(BaseModel):
    """Final response returned by /predict."""

    input_text: str
    prediction: PredictionResult


# -------------------- SIMILAR RESPONSE MODELS

class SimilarArticle(BaseModel):
    """One similar article."""

    rank: int
    article_id: str
    title: str
    category: str
    distance: float
    cosine_similarity: float


class SimilarResponse(BaseModel):
    """Final response returned by /similar."""

    input_text: str
    requested_results: int
    similar_articles_returned: int
    search_time_seconds: float
    similar_articles: list[SimilarArticle]


# -------------------- SHARED RESOURCES

shared_resources = {}


# -------------------- APPLICATION LIFESPAN

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared resources when the API starts."""

    # Load the embedding model once.
    embedding_model = load_embedding_model()

    # Open the existing ChromaDB collection once.
    (
        chroma_client,
        article_collection,
    ) = load_chroma_collection()

    # Store the embedding model.
    shared_resources["embedding_model"] = (
        embedding_model
    )

    # Store the ChromaDB client.
    shared_resources["chroma_client"] = (
        chroma_client
    )

    # Store the article collection.
    shared_resources["article_collection"] = (
        article_collection
    )

    # Store the total indexed article count.
    shared_resources["indexed_articles"] = (
        article_collection.count()
    )

    # API starts accepting requests here.
    yield

    # Remove references during shutdown.
    shared_resources.clear()


# -------------------- FASTAPI APPLICATION

app = FastAPI(
    title="AI News Intelligence API",
    description=(
        "News classification and "
        "semantic article search API."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# -------------------- HEALTH ENDPOINT

@app.get("/health")
def health_check():
    """Confirm that the API and resources are ready."""

    article_collection = shared_resources[
        "article_collection"
    ]

    return {
        "status": "ok",
        "service": "AI News Intelligence API",
        "embedding_model_loaded": (
            shared_resources.get(
                "embedding_model"
            )
            is not None
        ),
        "collection_name": (
            article_collection.name
        ),
        "indexed_articles": (
            shared_resources[
                "indexed_articles"
            ]
        ),
    }


# -------------------- PREDICTION ENDPOINT

@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict_news(
    request: PredictRequest,
):
    """Predict the category of news text."""

    # Reuse the existing ML prediction function.
    (
        class_index,
        category,
        confidence_percent,
    ) = predict_category_with_confidence(
        request.news_text
    )

    # Return JSON-friendly values.
    return {
        "input_text": request.news_text,
        "prediction": {
            "class_index": int(
                class_index
            ),
            "category": str(
                category
            ),
            "confidence_percent": float(
                confidence_percent
            ),
        },
    }


# -------------------- SIMILAR ARTICLE ENDPOINT

@app.post(
    "/similar",
    response_model=SimilarResponse,
)
def find_similar_articles(
    request: SimilarRequest,
):
    """Find semantically similar articles."""

    # Reuse the model loaded during startup.
    embedding_model = shared_resources[
        "embedding_model"
    ]

    # Reuse the ChromaDB collection.
    article_collection = shared_resources[
        "article_collection"
    ]

    # Prepare the query text.
    cleaned_text = prepare_query_text(
        request.news_text
    )

    # Convert the text into a query embedding.
    query_embedding = (
        generate_query_embedding(
            embedding_model,
            cleaned_text,
        )
    )

    # Search the existing ChromaDB collection.
    (
        raw_search_result,
        elapsed_seconds,
    ) = search_chroma_Collection(
        article_collection,
        query_embedding,
        n_results=request.top_k,
    )

    # Convert nested ChromaDB results
    # into clean article dictionaries.
    formatted_articles = (
        format_search_results(
            raw_search_result
        )
    )

    # Prepare the API response articles.
    similar_articles = []

    for article in formatted_articles:
        similar_articles.append(
            {
                "rank": int(
                    article["rank"]
                ),
                "article_id": str(
                    article["article_id"]
                ),
                "title": str(
                    article.get("title")
                    or "Untitled"
                ),
                "category": str(
                    article.get("label_name")
                    or "Unknown"
                ),
                "distance": float(
                    article["distance"]
                ),
                "cosine_similarity": float(
                    article[
                        "cosine_similarity"
                    ]
                ),
            }
        )

    # Return JSON-friendly search data.
    return {
        "input_text": cleaned_text,
        "requested_results": request.top_k,
        "similar_articles_returned": len(
            similar_articles
        ),
        "search_time_seconds": float(
            elapsed_seconds
        ),
        "similar_articles": similar_articles,
    }