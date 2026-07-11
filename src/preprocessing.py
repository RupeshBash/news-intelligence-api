def clean_text(text):
    """
    Clean a single text string.

    We do not lowercase here because TfidfVectorizer(lowercase=True)
    already handles lowercase during vectorization.
    """
    text = str(text)

    # Replace backslash with space
    text = text.replace("\\", " ")

    # Replace newline with space
    text = text.replace("\n", " ")

    # Remove extra spaces from start and end
    text = text.strip()

    return text


def combine_title_description(title, description):
    """
    Combine title and description into one clean text.
    """
    combined_text = str(title) + " " + str(description)
    cleaned_text = clean_text(combined_text)

    return cleaned_text


def prepare_text_column(df):
    """
    Create a clean text column from title and description columns.
    """
    df = df.copy()

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    df["text"] = df.apply(
        lambda row: combine_title_description(row["title"], row["description"]),
        axis=1
    )

    return df


if __name__ == "__main__":
    sample_text = "   APPLE\\AI\nNEWS   "

    print("Original:")
    print(repr(sample_text))

    print("Cleaned:")
    print(clean_text(sample_text))