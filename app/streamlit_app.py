# -------------------- IMPORTS AND PROJECT PATH

from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze_news import analyze_news
from src.search_articles import (
    load_embedding_model,
    load_chroma_collection,
)




#--------------------------CACHED ANALYSIS RESOURCES

@st.cache_resource
def load_analysis_resources():
    """Load MiniLM and ChromaDB once across Streamlit reruns."""

    embedding_model = load_embedding_model()

    (
        chroma_client,
        article_collection,
    ) = load_chroma_collection()

    return (
        embedding_model,
        chroma_client,
        article_collection,
    )




# -------------------- PAGE CONFIGURATION

st.set_page_config(
    page_title="AI News Intelligence",
    page_icon="📰",
    layout="centered",
)


# -------------------- PAGE HEADING

st.title("📰 AI News Intelligence")

st.write(
    "Enter a news headline or short article. "
    "The app will predict its category and retrieve "
    "semantically similar stored articles."
)


# -------------------- TEXT INPUT

user_text = st.text_area(
    label="News text",
    placeholder=(
        "Example: Apple announces new AI "
        "features for iPhone users"
    ),
    height=160,
)


# -------------------- ANALYZE BUTTON

analyze_button = st.button(
    "Analyze News",
    type="primary",
    use_container_width=True,
)


# -------------------- HANDLE ANALYSIS

if analyze_button:
    if not user_text.strip():
        st.warning(
            "Please enter a news headline or short article."
        )

    else:
        try:
            with st.spinner(
                "Analyzing the news and searching "
                "for similar articles..."
            ):
                (
                    embedding_model,
                    chroma_client,
                    article_collection,
                ) = load_analysis_resources()

                analysis_result = analyze_news(
                    news_text=user_text,
                    embedding_model=embedding_model,
                    article_collection=article_collection,
                    n_results=5,
                )

            st.subheader("Prediction Summary")

            (
                class_column,
                category_column,
                confidence_column,
            ) = st.columns(3)

            class_column.metric(
                label="Class index",
                value=str(
                    analysis_result[
                        "predicted_class_index"
                    ]
                ),
            )

            category_column.metric(
                label="Category",
                value=analysis_result[
                    "predicted_category"
                ],
            )

            confidence_column.metric(
                label="Confidence",
                value=(
                    f"{analysis_result[
                        'confidence_percent'
                    ]:.2f}%"
                ),
            )

            st.caption(
                "Similar articles returned: "
                f"{analysis_result[
                    'similar_articles_returned'
                ]} | "
                "Search time: "
                f"{analysis_result[
                    'search_time_seconds'
                ]:.4f} seconds"
            )

            # -------------------- SIMILAR ARTICLES

            st.subheader("Similar Articles")

            st.caption(
                "Lower cosine distance means a closer match. "
                "Higher cosine similarity means a closer match."
            )

            similar_articles = (
                analysis_result.get(
                    "similar_articles"
                )
                or []
            )

            if not similar_articles:
                st.info(
                    "No similar articles were found."
                )

            else:
                for position, article in enumerate(
                    similar_articles,
                    start=1,
                ):
                    rank = article.get(
                        "rank",
                        position,
                    )

                    title = (
                        article.get("title")
                        or "Untitled article"
                    )

                    category = (
                        article.get("label_name")
                        or "Unknown category"
                    )

                    article_id = (
                        article.get("article_id")
                        or "Unknown ID"
                    )

                    distance = article.get(
                        "distance"
                    )

                    similarity = article.get(
                        "cosine_similarity"
                    )

                    if distance is None:
                        distance_text = "Unavailable"
                    else:
                        distance_text = (
                            f"{float(distance):.4f}"
                        )

                    if similarity is None:
                        similarity_text = "Unavailable"
                    else:
                        similarity_text = (
                            f"{float(similarity):.4f}"
                        )

                    st.markdown(
                        f"### {rank}. {title}"
                    )

                    (
                        article_category_column,
                        distance_column,
                        similarity_column,
                    ) = st.columns(3)

                    article_category_column.metric(
                        label="Article category",
                        value=category,
                    )

                    distance_column.metric(
                        label="Cosine distance",
                        value=distance_text,
                    )

                    similarity_column.metric(
                        label="Cosine similarity",
                        value=similarity_text,
                    )

                    st.caption(
                        f"Article ID: {article_id}"
                    )

                    st.divider()



        except FileNotFoundError as error:
            st.error(
                "A required saved model or ChromaDB "
                f"resource was not found: {error}"
            )

        except ValueError as error:
            st.error(
                f"Invalid analysis data: {error}"
            )

        except Exception as error:
            st.error(
                f"News analysis failed: {error}"
            )

# -------------------- MODEL DETAILS

st.divider()

st.subheader("Model Details")

st.markdown(
    """
- **Model:** Logistic Regression
- **Features:** TF-IDF
- **Dataset:** AG News
- **Categories:** World, Sports, Business, Sci/Tech
"""
)