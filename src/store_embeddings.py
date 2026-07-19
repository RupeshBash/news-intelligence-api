#-------------------------IMPORTS

#Path creates OS-friendly file paths
from pathlib import Path

#ChromaDB stores embeddings, documents, and metadata
import chromadb

#NumPy loads the embedding matrix generated on Day 9
import numpy as np

#Pandas loads the matching article metadata
import pandas as pd



#-------------------------CONFIGURATION

#Day 9 generated files
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
    """Load and validate the Day 9 embeddings and metadata files."""

    #Stop early when the generated files are missing
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

    print("\nLoaded Day 9 files:")
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

    #Every metadata row must have one matching embedding row
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



#-----------------------------CREATE CHROMA COLLECTION

def create_chroma_collection():
    """Create or load the persistent AG News collection."""

    #Create the local database directory when needed
    CHROMA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nCreating persistent ChromaDB client...")
    print("ChromaDB path:", CHROMA_PATH)

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    #embedding_function=None is intentional because Day 9 already
    #generated the embeddings with all-MiniLM-L6-v2
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
        metadata={
            "description": "Balanced AG News article embeddings",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        },
    )

    print("Collection ready:", collection.name)
    print("Existing record count:", collection.count())

    return client, collection



#---------------PREPARE ALIGNED CHROMADB LISTS

def prepare_chroma_records(metadata_dataframe):
    """Prepare aligned IDs, documents, and metadata dictionaries."""

    #Unique article identifiers
    ids = metadata_dataframe["article_id"].astype(str).tolist()

    #Readable article text stored as each Chroma document
    documents = metadata_dataframe["text"].astype(str).tolist()

    #Extra searchable and display-related article information
    metadatas = []

    for _, row in metadata_dataframe.iterrows():
        article_metadata = {
            "source_row_id": int(row["source_row_id"]),
            "class_index": int(row["class_index"]),
            "label_name": str(row["label_name"]),
            "title": str(row["title"]),
        }

        metadatas.append(article_metadata)

    print("\nPrepared ChromaDB records:")
    print("ID count:", len(ids))
    print("Document count:", len(documents))
    print("Metadata count:", len(metadatas))

    return ids, documents, metadatas



#---------------------------VALIDATE CHROMA RECORDS

def validate_chroma_records(
    ids,
    documents,
    metadatas,
    embeddings,
):
    """Confirm that all values are ready for ChromaDB insertion."""

    record_count = len(ids)

    print("\nChromaDB record validation:")
    print("ID count:", len(ids))
    print("Document count:", len(documents))
    print("Metadata count:", len(metadatas))
    print("Embedding count:", embeddings.shape[0])

    if record_count != EXPECTED_RECORD_COUNT:
        raise ValueError(
            "Unexpected article count: "
            f"{record_count}"
        )

    if not (
        len(documents) == record_count
        and len(metadatas) == record_count
        and embeddings.shape[0] == record_count
    ):
        raise ValueError(
            "IDs, documents, metadatas, and embeddings "
            "do not have matching counts."
        )

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate article IDs were found."
        )

    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            "Unexpected embedding dimension: "
            f"{embeddings.shape[1]}"
        )

    print("ChromaDB record validation: Passed")



#------------------------INSERT CHROMA RECORDS

def insert_chroma_records(
    collection,
    ids,
    documents,
    metadatas,
    embeddings,
):
    """Insert prepared article records into ChromaDB in batches."""

    existing_count = collection.count()

    #Allow the script to be rerun without inserting duplicate IDs
    if existing_count == EXPECTED_RECORD_COUNT:
        print(
            "\nCollection already contains "
            f"{EXPECTED_RECORD_COUNT} records."
        )
        print("Insertion skipped.")

        return

    #A partially populated collection should be reviewed
    #instead of silently mixing old and new records
    if existing_count != 0:
        raise ValueError(
            "The collection is partially populated: "
            f"{existing_count} existing records. "
            "Expected either 0 or "
            f"{EXPECTED_RECORD_COUNT}."
        )

    print("\nInserting records into ChromaDB...")

    record_count = len(ids)

    for start_index in range(
        0,
        record_count,
        INSERT_BATCH_SIZE,
    ):
        end_index = min(
            start_index + INSERT_BATCH_SIZE,
            record_count,
        )

        #Convert the NumPy slice to normal Python lists
        embedding_batch = embeddings[
            start_index:end_index
        ].tolist()

        collection.add(
            ids=ids[start_index:end_index],
            documents=documents[start_index:end_index],
            metadatas=metadatas[start_index:end_index],
            embeddings=embedding_batch,
        )

        print(
            f"Inserted records "
            f"{start_index + 1} to {end_index}"
        )

    #Check the final count only after every batch is inserted
    final_count = collection.count()

    if final_count != EXPECTED_RECORD_COUNT:
        raise ValueError(
            "Unexpected collection count after insertion: "
            f"{final_count}"
        )

    print("ChromaDB insertion completed.")
    print("Stored record count:", final_count)



#----------------------------------VERIFY CHROMA COLLECTION

def verify_chroma_collection(
    collection,
    expected_first_id,
):
    """Verify the final count and retrieve one article by ID."""

    stored_count = collection.count()

    print("\nChromaDB collection verification:")
    print("Collection name:", collection.name)
    print("Stored record count:", stored_count)

    if stored_count != EXPECTED_RECORD_COUNT:
        raise ValueError(
            "Collection count validation failed: "
            f"{stored_count}"
        )

    #Retrieve one known record without similarity ranking
    result = collection.get(
        ids=[expected_first_id],
        include=[
            "documents",
            "metadatas",
        ],
    )

    if not result["ids"]:
        raise ValueError(
            f"Article was not found: {expected_first_id}"
        )

    print("\nRetrieved sample record:")
    print("ID:", result["ids"][0])
    print("Document:", result["documents"][0])
    print("Metadata:", result["metadatas"][0])
    print("Collection verification: Passed")



#----------------------MAIN WORKFLOW

def main():
    #Step 1: Load the Day 9 files
    metadata_dataframe, embeddings = load_saved_data()

    #Step 2: Create or load the persistent collection
    client, collection = create_chroma_collection()

    #Step 3: Prepare aligned ChromaDB lists
    ids, documents, metadatas = prepare_chroma_records(
        metadata_dataframe
    )

    #Step 4: Validate IDs, documents, metadata, and embeddings
    validate_chroma_records(
        ids,
        documents,
        metadatas,
        embeddings,
    )

    #Step 5: Insert all 1,000 records
    insert_chroma_records(
        collection,
        ids,
        documents,
        metadatas,
        embeddings,
    )

    #Step 6: Retrieve the first article and confirm persistence
    verify_chroma_collection(
        collection,
        ids[0],
    )

    print("\nDay 10 ChromaDB storage completed.")
    print("Database path:", CHROMA_PATH)
    print("Collection:", COLLECTION_NAME)
    print("Total stored articles:", collection.count())


if __name__ == "__main__":
    #Run using:
    #python -m src.store_embeddings
    main()