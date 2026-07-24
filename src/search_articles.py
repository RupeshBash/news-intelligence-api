#-------------------------IMPORTS

#Path creates os-friendly file paths
from pathlib import Path

#perf_counter measires vector-search execution time 
from time import perf_counter

#ChromaDB loads the persistent article collection
import chromadb

#SentenceTransformer creates the query embedding
from sentence_transformers import SentenceTransformer

#Reuse the existing project text-cleaning function
from src.preprocessing import clean_text



#---------------------------CONFIGURATION

#The same model used to generate the stored day9 embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

#Persistent ChromaDB storage creates on Day 10
CHROMA_PATH = Path("data/chroma_db")

#Existing collection containing 1000 AG News articles
COLLECTION_NAME = "ag_news_articles"

#Expected collection and embedding properties
EXPECTED_RECORD_COUNT = 1000
EXPECTED_EMBEDDING_DIMENSION = 384

#Number of similar articles returned for one query
DEFAULT_N_RESULTS = 5

#First semantic-search test query
SAMPLE_QUERY = (
    "A company introduced a new computer chip "
    "that uses less electricity."
)





#-----------------LOAD EMBEDDING MODEL

def load_embedding_model():
    """Load the pretrained model used for query embeddings."""

    print("Loading embedding model....")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    return model




#-----------------------LOAD CHROMA COLLECTION

def load_chroma_collection():
    """Load and validate the existing persistent collection."""

    #Stop early when the local database directory is missing
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB directory was not found: {CHROMA_PATH}"
        )

    print("\nLoading persistent chromaDB client....")
    print("ChromaDB path:", CHROMA_PATH)

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    #Load the existing collection instead of creating an empty one
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    stored_count = collection.count()

    print("Collection loaded:", collection.name)
    print("Stored article count:", stored_count)

    if stored_count != EXPECTED_RECORD_COUNT:
        raise ValueError(
            "Unexpected collection count: "
            f"{stored_count}"
        )
    print("Collection validation: Passed")

    return client, collection




#-------------------PREPARE QUERY TEXT

def prepare_query_text(query_text):
    """Clean and validate one semantic-search query."""

    if not isinstance(query_text, str):
        raise TypeError(
            "Query text must be a string."
        )

    cleaned_query = clean_text(query_text)

    if cleaned_query == "":
        raise ValueError(
            "Query text cannot be empty."
        )

    print("\nOriginal query:", query_text)
    print("Cleaned query:", cleaned_query)

    return cleaned_query



#-------------------------GENERATE QUERY EMBEDDING

def generate_query_embedding(
    model,
    query_text,
):
    """Generate and validate one normalized query embedding."""

    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("\nQuery embedding generated:")
    print("Embedding shape:", query_embedding.shape)
    print("Embedding dtype:", query_embedding.dtype)

    #One query should produce one one-dimensional vector
    if query_embedding.ndim != 1:
        raise ValueError(
            "Expected one 1D query embedding, got shape: "
            f"{query_embedding.shape}"
        )

    if query_embedding.shape[0] != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            "Unexpected query embedding dimension: "
            f"{query_embedding.shape[0]}"
        )

    print("Query embedding validation: Passed")

    return query_embedding




#-----------------------------SEARCH CHROMA COLLECTION

def search_chroma_Collection(
        collection,
        query_embedding,
        n_results=DEFAULT_N_RESULTS,
):
    """Search the collection using one precomputed query embedding."""

    if n_results <= 0:
        raise ValueError(
            "n_results must be greater than zero."
        )
    collection_count = collection.count()

    if n_results > collection_count:
        raise ValueError(
            "n_results cannot be greater than "
            f"the collection count: {collection_count}"
        )

    print("\nSearching ChromaDB...")
    print("Requested result count:", n_results)

    start_time = perf_counter()

    search_result = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    elapsed_seconds = perf_counter() - start_time

    print("Vector search completed.")
    print(f"Search time: {elapsed_seconds:.4f} seconds")

    return search_result, elapsed_seconds





#--------------------------FORMAT SEARCH RESULTS

def format_search_results(search_result):
    """Convert ChromaDB's nested output into clean ranked records."""

    id_groups = search_result.get("ids")

    if not id_groups or not id_groups[0]:
        return []

    document_groups = search_result.get("documents")
    metadata_groups = search_result.get("metadatas")
    distance_groups = search_result.get("distances")

    if document_groups is None:
        raise ValueError(
            "Search result documents are missing."
        )

    if metadata_groups is None:
        raise ValueError(
            "Search result metadata is missing."
        )

    if distance_groups is None:
        raise ValueError(
            "Search result distance are misssing."
        )

    ##Extract the result lists for the first and only query
    ids = id_groups[0]
    documents = document_groups[0]
    metadatas = metadata_groups[0]
    distances = distance_groups[0]

    result_count =len(ids)

    print("\nRaw search result counts:")
    print("ID count:", len(ids))
    print("Document count:", len(documents))
    print("Metadata count:", len(metadatas))
    print("Distance count:", len(distances))

    if not (
        len(documents) == result_count
        and len(metadatas) == result_count
        and len(distances) == result_count
    ):
        raise ValueError(
            "Search result lists do not have matching lengths."
        )

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate article IDs were returned."
        )

    print("Search result alignmet: Passed")
    print("Search result ID validation: Passed")

    ranked_results = []

    for index in range(result_count):
        metadata = metadatas[index]

        if metadata is None:
            metadata = {}

        distance = float(distances[index])

        #The collection uses cosine distance:
        #cosine similarity = 1 - cosine distance
        cosine_similarity = 1.0 - distance

        result_record = {
            "rank": index + 1,
            "article_id": ids[index],
            "title": metadata.get(
                "title",
                "unknown title",
            ),
            "label_name": metadata.get(
                "label_name",
                "unknown category",
            ),
            "class_index": metadata.get(
                "class_index",
            ),
            "source_row_id": metadata.get("source_row_id"),
            "document": documents[index],
            "distance": distance,
            "cosine_similarity": cosine_similarity,
        }

        ranked_results.append(result_record)

    return ranked_results





#---------------------------DISPLAY SEARCH RESULTS

def display_search_results(
        query_text,
        ranked_results,
        elapsed_seconds,
):
    """Print clean top-ranked article results."""

    print("\n" + "=" * 70)
    print("SIMILAR ARTICLE SEARCH")
    print("=" * 70)

    print("Query:", query_text)
    print("Results returned:", len(ranked_results))
    print(f"Vector search time: {elapsed_seconds:4f} seconds")

    if not ranked_results:
        print("\nNo similar articles were found.")
        return
    
    for result in ranked_results:
        print("\n" + "-" * 70)
        print(f"Rank {result['rank']}")
        print("Article ID:", result["article_id"])
        print("Title:", result["title"])
        print("Category:", result["label_name"])
        print("Class index:", result["class_index"])
        print("Source row ID:", result["source_row_id"])
        print(f"Distance: {result['distance']:.4f}")
        print(
            "Cosine similarity:",
            f"{result['cosine_similarity']:.4f}",
        )
        print("Document:", result["document"])

    best_result =ranked_results[0]

    print("\n" + "=" * 70)
    print("BEST MATCH")
    print("=" * 70)
    print("Title:", best_result["title"])
    print("Category:", best_result["label_name"])
    print(f"Distance: {best_result['distance']:.4f}")






#------------------------------MAIN WORKFLOW

def main():
    #Step 1: Load the same model used for stored embeddings
    model = load_embedding_model()

    #Step 2: Load the existing persistent chroma collection
    client, collection = load_chroma_collection()

    #Step 3: clean and validate the sample search query
    cleaned_query = prepare_query_text(
        SAMPLE_QUERY
    )

    #Step 4: convert the query into one 384-dimensional vector
    query_embedding = generate_query_embedding(
        model,
        cleaned_query,
    )

    #Step 5: search for the five closest stored articles
    search_result, elapsed_seconds = search_chroma_Collection(
        collection, 
        query_embedding, 
        n_results=DEFAULT_N_RESULTS,
    )

    #Step 6: Convert chroma's nested response into clean records
    ranked_results = format_search_results(
        search_result
    )

    #Step 7: Display the ranked results
    display_search_results(
        cleaned_query,
        ranked_results,
        elapsed_seconds,
    )

    print("\nDay 11 similar-article search completed.")


if __name__ == "__main__":
    #Run using:
    #python -m src.search_articles
    main()

