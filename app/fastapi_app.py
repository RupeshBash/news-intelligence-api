#---------------------------------IMPORTS

from contextlib import asynccontextmanager

from fastapi import FastAPI 

from src.search_articles import (
    load_embedding_model,
    load_chroma_collection,
)



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

