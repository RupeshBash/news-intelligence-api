#---------------------IMPORTS

# Path creates OS-friendly file paths
from pathlib import Path

# perf_counter measures embedding-generation time
from time import perf_counter

#NumPy stores and saves the embedding matrix
import numpy as np

#Pandas loads and prepares the AG News records
import pandas as pd

#SentenceTransformer loads the pretrained embedding model
from sentence_transformers import SentenceTransformer

#Reuse the existing project text-cleaning function
from src.preprocessing import clean_text


#----------------------CONFIGURATION

#Pretrained model used to create semantic text embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DATA_PATH = Path("data/raw/train.csv")

#Generated files will be stored here
OUTPUT_DIR = Path("data/processed")

EMBEDDINGS_PATH = OUTPUT_DIR / "article_embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "article_metadata.csv"

#Use 250 articles from each of the four categories
ROWS_PER_CLASS = 250

# 250 rows * 4 categories = 1000 total articles
TOTAL_SAMPLE_SIZE = ROWS_PER_CLASS * 4

# Number of texts sent to the model at one time
BATCH_SIZE = 32

# keep sampling reproducible
RANDOM_STATE = 42

# all-MiniLM-L6-v2 produces 384 values for each text
EXPECTED_EMBEDDING_DIMENSION = 384


LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}



#--------------------------------LOAD AND PREPARE DATA

def load_and_prepare_articles(data_path):
    """Load AG News and create clean article text."""

    #Give a clear error when the configured CSV path is wrong
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {data_path}"
        )

    print("Loading dataset from:", data_path)

    dataframe = pd.read_csv(data_path)

    print("Original dataset shape:", dataframe.shape)
    print("Original columns:", dataframe.columns.tolist())

    #Support both the original Kaggle column names
    #and the cleaned names used in this project
    dataframe = dataframe.rename(
        columns={
            "Class Index": "class_index",
            "Title": "title",
            "Description": "description",
        }
    )

    required_columns = {
        "class_index",
        "title",
        "description",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )
    
    #Preserve the original CSV row position
    #This gives each article a reproducible unique ID later
    dataframe["source_row_id"] = dataframe.index

    #Replace missing title or description values with empty strings
    dataframe["title"] = dataframe["title"].fillna("")
    dataframe["description"] = dataframe["description"].fillna("")

    #combine title and description into one article text
    dataframe["text"] = (
        dataframe["title"]
        + " "
        + dataframe["description"]
    )

    #Apply the project's existing light cleaning function
    dataframe["text"] = dataframe["text"].map(clean_text)

    #convert numeric labels into readable category names
    dataframe["label_name"] = dataframe["class_index"].map(
        LABEL_MAP
    )

    #Remove empty text and unknown labels
    dataframe = dataframe[
        (dataframe["text"] != "")
        & (dataframe["label_name"].notna())
    ].copy()
    
    print("Usable dataset shape:", dataframe.shape)

    return dataframe



#---------------------------create balanced sample

def create_balanced_sample(dataframe):
    """Select the same number of articles from every category."""

    print("\nAvailable rows per category:")

    # First, check that every category has enough rows.
    for class_id, label_name in LABEL_MAP.items():
        class_count = (
            dataframe["class_index"] == class_id
        ).sum()

        print(f"{label_name}: {class_count}")

        # Stop when the dataset cannot provide enough rows.
        if class_count < ROWS_PER_CLASS:
            raise ValueError(
                f"{label_name} has only {class_count} rows. "
                f"Required: {ROWS_PER_CLASS}"
            )

    # Select 250 random records from each category.
    sampled_dataframe = (
        dataframe
        .groupby(
            "class_index",
            group_keys=False,
        )
        .sample(
            n=ROWS_PER_CLASS,
            random_state=RANDOM_STATE,
        )
    )

    # Shuffle the complete sample after combining all categories.
    sampled_dataframe = sampled_dataframe.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    # Create a unique article ID from the original source-row index.
    sampled_dataframe["article_id"] = (
        sampled_dataframe["source_row_id"]
        .map(lambda row_id: f"ag_news_train_{row_id}")
    )

    # Keep only the information needed by later search stages.
    sampled_dataframe = sampled_dataframe[
        [
            "article_id",
            "source_row_id",
            "class_index",
            "label_name",
            "title",
            "description",
            "text",
        ]
    ]

    print("\nBalanced sample shape:", sampled_dataframe.shape)

    print("Balanced sample counts:")

    for class_id, label_name in LABEL_MAP.items():
        sample_count = (
            sampled_dataframe["class_index"] == class_id
        ).sum()

        print(f"{label_name}: {sample_count}")

    return sampled_dataframe



#---------------------Generate Embeddings

def generate_article_embeddings(sample_dataframe):
    """Generate one embedding for every sampled article."""

    print("\nLoading embedding model.....")

    #Load the pretrained model once.
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    # Convert the pandas text column into a normal python list
    article_texts = sample_dataframe["text"].tolist()

    print("Articles to embed:", len(article_texts))
    print("Batch size:", BATCH_SIZE)

    start_time = perf_counter()

    #encode() generates one embedding for each article text
    #
    #batch_size controls how many texts are processed together
    #show_progress_bar displays generation progress
    #convert_to_numpy returns a NumPy matrix
    #normalize_embeddings gives every vector consistent unit length
    embeddings = model.encode(
        article_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    elapsed_seconds = perf_counter() - start_time

    print("Embedding generation completed.")
    print(f"Generation time: {elapsed_seconds:.2f} seconds")

    return embeddings, elapsed_seconds



#---------------Validate Generated Data

def validate_generated_data(sample_dataframe, embeddings):
    """Verify that article metadata and embeddings remain aligned."""

    article_count = len(sample_dataframe)
    embedding_count = embeddings.shape[0]

    #Every article must have exactly one embedding
    if article_count != embedding_count:
        raise ValueError(
            "Article and embedding counts do not match: "
            f"{article_count} articles, "
            f"{embedding_count} embeddings"
        )
    
    #The embedding result should be a two-dimensional matrix
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got {embeddings.ndim}D."
        )
    
    #Verify the expected model dimension
    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            "Unexpected embedding dimension: "
            f"{embeddings.shape[1]}"
        )
    
    #ChromaDB will require unique document IDs later
    if not sample_dataframe["article_id"].is_unique:
        raise ValueError("Duplicate article IDs were found.")
    
    print("\nValidation passed:")
    print("Article count:", article_count)
    print("Embedding count:", embedding_count)
    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    print("Article IDs unique: Yes")



#------------------Save Outputs

def save_generated_data(sample_dataframe, embeddings):
    """Save embeddings and their matching article metadata."""

    #Create data/processed when it does not already exist
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    #Save the numeric matrix in NumPy's binary .npy format
    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    #save article IDs, labels, titles, and text separately
    sample_dataframe.to_csv(
        METADATA_PATH,
        index=False,
    )

    print("\nSaved generated files:")
    print("Embeddings:", EMBEDDINGS_PATH)
    print("Metadata:", METADATA_PATH)



#-----------------Verify Saved Files

def verify_saved_files():
    """Load the generated files again and confirm their shapes."""

    loaded_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    loaded_metadata = pd.read_csv(
        METADATA_PATH
    )

    print("\nSaved-file verification:")
    print(
        "Loaded embedding shape:",
        loaded_embeddings.shape,
    )
    print(
        "Loaded metadata shape:",
        loaded_metadata.shape,
    )

    if len(loaded_metadata) != loaded_embeddings.shape[0]:
        raise ValueError(
            "Saved metadata and embeddings are misaligned."
        )

    print("Saved-file alignment: Correct")



# ---------------- MAIN WORKFLOW

def main():
    #step1: Load and clean the complete AG News training data
    articles_dataframe = load_and_prepare_articles(
        DATA_PATH
    )

    #Step2: select 250 articles from every category
    sample_dataframe = create_balanced_sample(
        articles_dataframe
    )

    print("\nFirst sampled article:")
    print("Article ID:", sample_dataframe.iloc[0]["article_id"])
    print("Category:", sample_dataframe.iloc[0]["label_name"])
    print("Text:", sample_dataframe.iloc[0]["text"])

    #step3: Generate one embedding for each sampled article
    embeddings, elapsed_seconds = generate_article_embeddings(
        sample_dataframe
    )

    #step4: confirm metadata and embeddings are aligned
    validate_generated_data(
        sample_dataframe,
        embeddings,
    )

    #step 5: save the two generated files
    save_generated_data(
        sample_dataframe,
        embeddings,
    )
    
    #step 6: Load the files again to confirm they were saved correctly 
    verify_saved_files()

    print("\nDay 9 embedding generation completed.")
    print("Total sampled articles:", len(sample_dataframe))
    print("Embedding matrix shape:", embeddings.shape)
    print(f"Total generation time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    #run using 
    #python -m src.generate_embeddings
    main()