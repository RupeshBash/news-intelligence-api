# -------------- imports and model constant

# SentenceTransformer loads the pretrained embedding model.
# util provides helper functions like cosine similarity.
from sentence_transformers import SentenceTransformer, util

# Reuse the same cleaning function from our classifier project.
from src.preprocessing import clean_text


# Store the model name once so we do not repeat a long string many times.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ----------- load the embedding model

def load_embedding_model():
    """Load the pretrained text-embedding model."""

    print("Loading embedding model...")

    # This loads a pretrained model from SentenceTransformers.
    # It does NOT train or fine-tune the model.
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    # Return the model so we can use it in other functions.
    return model


# ----------- generate one embedding

def generate_embedding(model, text):
    """Clean one text and convert it into one embedding."""

    # Clean the input text using our existing project cleaning function.
    cleaned_text = clean_text(text)

    # Do not send empty text to the embedding model.
    if cleaned_text == "":
        raise ValueError("Text cannot be empty.")

    # encode() converts text into a numeric vector.
    # For one string, it returns one embedding.
    embedding = model.encode(cleaned_text)

    return embedding


# ----------- main function

def main():
    # Load the model once.
    # We reuse the same model for the single embedding
    # and the similarity comparison.
    model = load_embedding_model()

    # ------------------ single embedding test

    sample_text = (
        "The company introduced a smaller processor "
        "that uses less power."
    )

    # Convert one sample sentence into one embedding.
    sample_embedding = generate_embedding(
        model,
        sample_text,
    )

    print("\nSingle embedding result:")
    print("Input text:", sample_text)
    print("Embedding type:", type(sample_embedding))
    print("Embedding shape:", sample_embedding.shape)

    # Printing all 384 numbers would make the terminal messy.
    # Therefore, print only the first 10 values.
    print("First 10 values:", sample_embedding[:10])

    # ------------------ compare three sentences

    # The first sentence is our query.
    query_text = (
        "The company introduced a smaller processor "
        "that uses less power."
    )

    # This sentence has related meaning but uses different words.
    related_text = (
        "A new chip design improves performance "
        "while reducing energy use."
    )

    # This sentence is about a different topic.
    unrelated_text = (
        "Representatives met to discuss an agreement "
        "after several weeks of tension."
    )

    # Store all three texts in a list.
    # The index order matters because embeddings follow the same order.
    texts = [
        query_text,        # index 0
        related_text,      # index 1
        unrelated_text,    # index 2
    ]

    # ------------------ clean all three sentences

    cleaned_texts = []

    for text in texts:
        # Clean each sentence one by one.
        cleaned_text = clean_text(text)

        # Stop early if any sentence becomes empty.
        if cleaned_text == "":
            raise ValueError("Comparison text cannot be empty.")

        # Add the cleaned sentence to the list.
        cleaned_texts.append(cleaned_text)

    # ------------------ generate batch embeddings

    # Passing a list creates one embedding for every text.
    # convert_to_tensor=True returns a PyTorch tensor.
    embeddings = model.encode(
        cleaned_texts,
        convert_to_tensor=True,
    )

    print("\nBatch embedding result:")
    print("Number of texts:", len(cleaned_texts))
    print("Embeddings shape:", embeddings.shape)

    # ------------------ select each embedding by index

    # embeddings[0] belongs to query_text.
    query_embedding = embeddings[0]

    # embeddings[1] belongs to related_text.
    related_embedding = embeddings[1]

    # embeddings[2] belongs to unrelated_text.
    unrelated_embedding = embeddings[2]

    # ------------------ calculate cosine similarity

    # Compare the query sentence with the related sentence.
    related_similarity = util.cos_sim(
        query_embedding,
        related_embedding,
    ).item()

    # Compare the same query with the unrelated sentence.
    unrelated_similarity = util.cos_sim(
        query_embedding,
        unrelated_embedding,
    ).item()

    # .item() extracts one normal Python number
    # from the single-value tensor returned by cos_sim().

    # ------------------ print similarity results

    print("\nCosine similarity results:")

    print("\nQuery:")
    print(query_text)

    print("\nRelated text:")
    print(related_text)
    print(f"Similarity: {related_similarity:.4f}")

    print("\nUnrelated text:")
    print(unrelated_text)
    print(f"Similarity: {unrelated_similarity:.4f}")

    # ------------------ choose the most similar sentence

    # The larger cosine score is treated as the closer result.
    if related_similarity > unrelated_similarity:
        most_similar_text = related_text
        highest_similarity = related_similarity
    else:
        most_similar_text = unrelated_text
        highest_similarity = unrelated_similarity

    print("\nMost similar result:")
    print(most_similar_text)
    print(f"Similarity score: {highest_similarity:.4f}")


# ----------- run the file

if __name__ == "__main__":
    # This runs main() only when we execute:
    # python -m src.embedding_demo
    main()