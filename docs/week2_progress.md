# Week 2 Development Progress

This document contains the day-by-day development record for the semantic
embedding and similar-article search stage of the project.

## Week 2 Goal

Add semantic article search using SentenceTransformer embeddings and ChromaDB.

```text
News text
    ↓
SentenceTransformer
    ↓
Semantic embedding
    ↓
ChromaDB vector search
    ↓
Top similar articles
```

---

## Day 8 Progress — Introduction to Text Embeddings

### What I completed

- Installed the `sentence-transformers` package
- Loaded the pretrained `all-MiniLM-L6-v2` model
- Generated an embedding for one news sentence
- Confirmed that one embedding contains 384 numeric values
- Generated embeddings for three sentences as one batch
- Compared related and unrelated sentences using cosine similarity
- Confirmed that the related sentence received the higher similarity score
- Reused the existing `clean_text()` preprocessing function
- Created `src/embedding_demo.py`
- Added input validation for empty text

### Files created or updated

```text
src/embedding_demo.py
requirements.txt
docs/week2_progress.md
README.md
```

### Day 8 Workflow

```text
Raw news text
        ↓
clean_text()
        ↓
all-MiniLM-L6-v2
        ↓
384-dimensional embedding
        ↓
Cosine similarity
        ↓
Most similar sentence
```

### Example Texts

Query:

```text
The company introduced a smaller processor that uses less power.
```

Related text:

```text
A new chip design improves performance while reducing energy use.
```

Unrelated text:

```text
Representatives met to discuss an agreement after several weeks of tension.
```

### Actual Result

```text
Single embedding type: NumPy array
Single embedding shape: (384,)

Number of comparison texts: 3
Batch embedding shape: torch.Size([3, 384])

Related similarity: 0.5342
Unrelated similarity: -0.0250

Most similar result:
A new chip design improves performance while reducing energy use.
```

The related sentence received a higher cosine-similarity score than the
unrelated sentence:

```text
0.5342 > -0.0250
```

This means the embedding model correctly ranked the related sentence as the
closer semantic match.

### What is a Text Embedding?

A text embedding is a numeric representation of text.

The embedding model receives text such as:

```text
A new chip design reduces energy usage.
```

It converts the text into a vector containing numbers:

```text
[0.0350, 0.1048, -0.0153, ...]
```

The `all-MiniLM-L6-v2` model creates a vector containing 384 values for each
input sentence.

These values are not category IDs or individual word counts. Together, they
represent patterns learned by the pretrained model.

### What Does the Embedding Shape Mean?

For one sentence:

```text
(384,)
```

This means:

```text
one sentence
384 embedding dimensions
```

For three sentences:

```text
torch.Size([3, 384])
```

This means:

```text
3 sentences × 384 values for each sentence
```

### What is Cosine Similarity?

Cosine similarity compares the direction of two vectors.

General interpretation:

```text
Score closer to 1  → stronger similarity
Score closer to 0  → weak or no similarity
Negative score     → vectors point in different directions
```

The score is produced by the embedding model and should not be treated as a
guaranteed truth label.

The important result in this experiment was:

```text
related similarity > unrelated similarity
```

### TF-IDF Versus SentenceTransformer Embeddings

#### TF-IDF

TF-IDF represents text mainly using:

- words in the training vocabulary
- word frequency
- word importance
- sparse numeric features

It is useful for news classification because Logistic Regression can learn
which word features are important for each category.

#### SentenceTransformer Embeddings

SentenceTransformer represents text using:

- dense numeric vectors
- pretrained language patterns
- semantic relationships
- sentence-level meaning

This makes it useful for finding related articles even when they do not use
exactly the same words.

Example:

```text
smaller processor using less power
```

and:

```text
chip design reducing energy use
```

use different wording but express related meaning.

### Main Functions

#### `load_embedding_model()`

Loads the pretrained SentenceTransformer model.

```python
model = SentenceTransformer(MODEL_NAME)
```

This loads an already trained model.

It does not train or fine-tune a new model.

#### `generate_embedding()`

This function:

1. Receives the model and input text.
2. Cleans the text.
3. Validates that the text is not empty.
4. Calls `model.encode()`.
5. Returns the generated embedding.

```text
text
  ↓
clean_text()
  ↓
validation
  ↓
model.encode()
  ↓
embedding
```

#### `main()`

The `main()` function:

- loads the model once
- creates one sample embedding
- creates a batch of three embeddings
- selects embeddings by index
- calculates cosine similarity
- prints the most similar result

### Important Code Concepts

#### `model.encode()`

```python
embedding = model.encode(cleaned_text)
```

Converts text into an embedding.

#### Batch encoding

```python
embeddings = model.encode(
    cleaned_texts,
    convert_to_tensor=True,
)
```

Creates embeddings for several texts together.

#### Selecting an embedding

```python
query_embedding = embeddings[0]
related_embedding = embeddings[1]
unrelated_embedding = embeddings[2]
```

Each embedding keeps the same position as its original text in the list.

#### Cosine similarity

```python
related_similarity = util.cos_sim(
    query_embedding,
    related_embedding,
).item()
```

`util.cos_sim()` calculates similarity between two embeddings.

`.item()` extracts the single numeric value from the returned tensor.

### Validation

The embedding function checks for empty input:

```python
if cleaned_text == "":
    raise ValueError("Text cannot be empty.")
```

This prevents meaningless empty text from being sent to the embedding model.

### Command Used

From the project root:

```powershell
python -m src.embedding_demo
```

### Warning Observed

The terminal showed:

```text
You are sending unauthenticated requests to the HF Hub.
```

This was only a warning.

The model downloaded and loaded successfully, so an Hugging Face token was not
required for this small local experiment.

### Mistake Pattern

The file initially contained a duplicated similarity-calculation section.

The duplicated section also introduced misspelled variable names such as:

```text
realted_similarity
unrealted_embedding
related_embeddings
```

This caused a `NameError`.

The fix was to:

- remove the duplicated block
- keep only one similarity-calculation flow
- use consistent variable names

### Main Insight

The manual vector exercise and the real embedding experiment used the same
general idea:

```text
Manual practice
small vectors
    ↓
manual cosine-similarity calculation

Real project
384-dimensional embeddings
    ↓
SentenceTransformers cosine similarity
```

The difference is that the real embeddings were generated by a pretrained
language model.

### Current Limitation

Day 8 compared only three sentences stored temporarily in memory.

The project has not yet:

- generated embeddings for an AG News dataset sample
- saved article embeddings
- stored articles in ChromaDB
- created a reusable similar-article search function
- returned the top five similar articles

These tasks will be completed during the next Week 2 stages.

### Day 8 Final Takeaway

A SentenceTransformer converts text into a dense numeric embedding.

Cosine similarity compares embedding directions.

In this experiment, the related sentence received a higher score than the
unrelated sentence, showing that the pretrained model captured semantic
relationships between differently worded sentences.