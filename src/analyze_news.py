#---------------------------------IMPORTS

#Existing trained news classifier
from src.predict import (
    predict_category_with_confidence
)

#Existing day 11 semantic-search functions
from src.search_articles import (
    DEFAULT_N_RESULTS,
    load_embedding_model,
    load_chroma_collection,
    prepare_query_text,
    generate_query_embedding,
    search_chroma_Collection,
    format_search_results,
)



#------------------------CONFIGURATION

SAMPLE_NEWS_TEXT = (
    "A company introduced a new computer chip "
    "that uses less electricity."
)





#---------------VALIDATE INPUT

def validate_news_text(news_text):
    """Validate one news headline or article."""

    if not isinstance(news_text, str):
        raise TypeError(
            "News text must be a string."
        )
    cleaned_whitespace = " ".join(
        news_text.split()
    )

    if cleaned_whitespace == "":
        raise ValueError(
            "News text cannot be empty."
        )

    return cleaned_whitespace





#---------------------ANALYZE NEWS

def analyze_news(
        news_text,
        embedding_model,
        article_collection,
        n_results=DEFAULT_N_RESULTS,
):
    """
    Predict the news category and find similar articles.
    """

    # Validate before using either model
    validated_text = validate_news_text(
        news_text
    )

    if not isinstance(n_results, int):
        raise TypeError(
            "n_results must be an integer."
        )

    if n_results <= 0:
        raise ValueError(
            "n_results must be greater than zero."
        )

    #Step 1: Classify the news text
    (
        predicted_class_index,
        predicted_category,
        confidence_percent,
    ) = predict_category_with_confidence(
    validated_text
    )

    # Step 2: Prepare the text for semantic search
    cleaned_query = prepare_query_text(
        validated_text
    )

    #Step 3: Generate the query embedding
    query_embedding = generate_query_embedding(
        embedding_model,
        cleaned_query,
    )

    #Step 4: Search ChromaDB using the embedding
    (
        raw_search_result,
        elapsed_seconds,
    ) = search_chroma_Collection(
        article_collection,
        query_embedding,
        n_results=n_results,
    )

    #Step 5: Convert ChromaDB's nested result
    similar_articles = format_search_results(
        raw_search_result
    )

    #Convert Numpy values into normal python values
    analysis_result = {
        "input_text": validated_text,
        "predicted_class_index": int(
            predicted_class_index
        ),
        "predicted_category": predicted_category,
        "confidence_percent": float(
            confidence_percent
        ),
        "similar_articles_returned": len(
            similar_articles
        ),
        "search_time_seconds": round(
            elapsed_seconds,
            6,
        ),
        "similar_articles": similar_articles,
    }

    return analysis_result




# ------------------------- DISPLAY RESULT

def display_analysis_result(analysis_result):
    """Display combined classification and search results."""

    print("\n" + "=" * 70)
    print("AI NEWS ANALYSIS")
    print("=" * 70)

    print(
        "Input text:",
        analysis_result["input_text"],
    )

    print(
        "Predicted class index:",
        analysis_result[
            "predicted_class_index"
        ],
    )

    print(
        "Predicted category:",
        analysis_result[
            "predicted_category"
        ],
    )

    print(
        "Prediction confidence:",
        f"{analysis_result['confidence_percent']:.2f}%",
    )

    print(
        "Similar articles returned:",
        analysis_result[
            "similar_articles_returned"
        ],
    )

    print(
        "Search time:",
        f"{analysis_result['search_time_seconds']:.4f}",
        "seconds",
    )

    similar_articles = analysis_result[
        "similar_articles"
    ]

    if not similar_articles:
        print("\nNo similar articles were found.")
        return

    print("\n" + "=" * 70)
    print("SIMILAR ARTICLES")
    print("=" * 70)

    for article in similar_articles:
        print("\n" + "-" * 70)
        print("Rank:", article["rank"])
        print(
            "Article ID:",
            article["article_id"],
        )
        print(
            "Title:",
            article["title"],
        )
        print(
            "Category:",
            article["label_name"],
        )
        print(
            "Distance:",
            f"{article['distance']:.4f}",
        )
        print(
            "Cosine similarity:",
            f"{article['cosine_similarity']:.4f}",
        )

    # This block must be outside the for-loop
    best_match = similar_articles[0]

    print("\n" + "=" * 70)
    print("BEST SIMILAR ARTICLE")
    print("=" * 70)
    print(
        "Title:",
        best_match["title"],
    )
    print(
        "Category:",
        best_match["label_name"],
    )
    print(
        "Distance:",
        f"{best_match['distance']:.4f}",
    )




#--------------------MAIN WORKFLOW

def main():
    #Load the embedding model once
    embedding_model = load_embedding_model()

    #Load the persistent ChromaDB collection once
    (
        chroma_client,
        article_collection,
    ) = load_chroma_collection()

    analysis_result = analyze_news(
        news_text=SAMPLE_NEWS_TEXT,
        embedding_model=embedding_model,
        article_collection=article_collection,
        n_results=5, 
    )

    display_analysis_result(
        analysis_result
    )

    print(
        "\nDay 12 combined news analysis completed."
    )


if __name__ == "__main__":
    main()
