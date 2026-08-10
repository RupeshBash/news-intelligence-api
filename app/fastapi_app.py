# -------------------- IMPORTS

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from src.predict import (
    predict_category_with_confidence,
)

from src.search_articles import (
    load_embedding_model,
    load_chroma_collection,
)




#---------------------------------REQUEST MODELS

class PredictRequest(BaseModel):
    """Expected JSON body for prediction."""

    #client must send at least one character
    news_text: str = Field(
        min_length=1,
        max_length=5000,
    )

    @field_validator("news_text")
    @classmethod
    def clean_news_text(cls, value):
        """Clean and reject empty text."""

        #Remove repeated spaces and line breaks
        cleaned_text = " ".join(
            value.split()
        )

        #Reject whitespace-only input
        if not cleaned_text:
            raise ValueError(
                "news_text cannot be empty."
            )

        return cleaned_text




#--------------------------RESPONSE MODELS

class PredictionResult(BaseModel):
    """Prediction details."""

    class_index: int
    category: str
    confidence_percent: float


class PredictResponse(BaseModel):
    """Final response returned by /predict."""

    input_text: str
    prediction: PredictionResult


#--------------------------------SHARED RESOURCES

shared_resources = {}


#---------------------------------APPLICATION LIFESPAN

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared resources when the API starts."""

    # Load the embedding model once
    embedding_model = load_embedding_model()

    # open the existing chromaDB collection once
    (
        chroma_client,
        article_collection,
    ) = load_chroma_collection()

    #store resources for future endpoints.
    shared_resources["embedding_model"] = (
        embedding_model
    )

    shared_resources["chroma_client"] = (
        chroma_client
    )

    shared_resources["article_collection"] = (
    article_collection
    )

    shared_resources["indexed_articles"] = (
        article_collection.count()
    )

    #API starts accepting request here
    yield

    #Reomve references during shutdown
    shared_resources.clear()




#-----------------------------FASTAPI APPLICATION

app = FastAPI(
    title="AI News Intelligence API",
    description=(
        "News classification and "
        "semantic article search API."
    ),
    version="0.1.0",
    lifespan=lifespan,
)





#-------------------------HEALTH ENDPOINT

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





#----------------------PREDICTION ENDPOINT

@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict_news(
    request: PredictRequest,
):
    """Predict the category of news text."""

    #Reuse the existing ML prediction function
    (
        class_index,
        category,
        confidence_percent,
    ) = predict_category_with_confidence(
        request.news_text
    )

    # Return normal JSON-friendly values
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