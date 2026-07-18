#-------------------------IMPORTS

#Path created OS-friendly file paths
from pathlib import Path

#ChromaDB stores embeddings, documents, and metadata
import chromadb

#NumPy loads the embedding matrix generated on Day 9
import numpy as np

#Pandas loads the matching article metadata
import pandas as pd



#-------------------------CONFIGURATION

#day9 generated files
EMBEDDINGS_PATH = Path("data/processed/article_embeddings.npy")
METADATA_PATH = Path("data/processed/article_metadata.csv")

#ChromaDB will store its persistent database files here
CHROMA_PATH = Path("data/chroma_db")

#Name of the article collection inside ChromaDB
COLLECTION_NAME = "ag_news_articles"

#Insert smaller groups instead of sending all records at once
INSERT_BATCH_SIZE = 100

#Expected Day 9 results
EXPECTED_RECORD_COUNT = 1000
EXPECTED_EMBEDDING_DIMENSION = 384

#Model that originally created the saved embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



#--------------------------LOAD SAVED DATA

def load_saved_data():
    """Load and validate the day 9 embeddings and metadata files."""

    #stop early when the generated files are missing
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embedding file was not found: {EMBEDDINGS_PATH}"
        )
    
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file was not found: {METADATA_PATH}"
        )
    
    print("Loading embeddings from:", EMBEDDINGS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    print("Loading metadata from:", METADATA_PATH)
    metadata_dataframe = pd.read_csv(METADATA_PATH)
    
    print("\nLoaded day 9 files:")
    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    print("Metadata shape:", metadata_dataframe.shape)

    required_columns = {
        "article_id",
        "source_row_id",
        "class_index",
        "label_name",
        "title",
        "text",
    }

    missing_columns = (
        required_columns
        - set(metadata_dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: {missing_columns}"
        )
    
    if len(metadata_dataframe) != embeddings.shape[0]:
        raise ValueError(
            "Metadata and embedding counts do not match: "
            f"{len(metadata_dataframe)} metadata rows, "
            f"{embeddings.shape[0]} embedding rows"
        )
    
    #The embedding matrix must contain rows and columns
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got {embeddings.ndim}D."
        )
    
    #Verify the expected MiniLM embedding dimension
    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            "Unexpected embedding dimension: "
            f"{embeddings.shape[1]}"
        )
    
    #ChromaDB requires a unique ID for every record
    if not metadata_dataframe["article_id"].is_unique:
        raise ValueError(
            "Duplicate article IDs were found in the metadata."
        )

    #Prevent null values in the fields used for database insertion
    if metadata_dataframe[list(required_columns)].isnull().any().any():
        raise ValueError(
            "Null values were found in required metadata columns."
        )

    print("Saved-data validation: Passed")
    print("Article IDs unique: Yes")

    return metadata_dataframe, embeddings











