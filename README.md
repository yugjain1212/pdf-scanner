# PDF Page Similarity Search

A small command-line information-retrieval tool that finds the page in a
selectable-text PDF most similar to a user-provided query. PDF files are local
inputs and are intentionally excluded from version control. Place your PDF at
`data/document.pdf` or pass another PDF path to the command.

## Problem statement

Given a PDF and a short text query, identify the original PDF page whose text
is most relevant to the query, and show its similarity score and page text.

## Objective

This program demonstrates a classical NLP / information-retrieval pipeline:
extract each PDF page separately, clean the text, represent pages and a query
with TF-IDF, then rank pages by cosine similarity. It does not use an LLM,
embeddings, or a vector database.

## Technologies

- Python 3.10+
- PyMuPDF — selectable text extraction
- scikit-learn — TF-IDF vectorization and cosine similarity
- NumPy — stable ranking of similarity scores

## Algorithm

```text
PDF
 ↓
Page-wise text extraction
 ↓
Text cleaning
 ↓
TF-IDF page vectors
 ↓
Query vectorization using the same fitted vectorizer
 ↓
Cosine similarity
 ↓
Similarity ranking
 ↓
Most relevant original PDF page
```

Each non-empty page is one document in the TF-IDF collection. The vectorizer is
fitted once when `PDFSearchEngine` is created. Every query is transformed with
that same vocabulary; the page vectors are not refitted per query. Empty pages
are not indexed, but the original PDF page numbers are kept.

### Why TF-IDF?

- **TF (term frequency)** measures how often a word occurs in a page.
- **IDF (inverse document frequency)** gives less weight to words that occur on
  many pages and more weight to words that are relatively rare.
- **TF-IDF = TF × IDF** balances those two ideas. Common English stop words are
  removed using `TfidfVectorizer(stop_words="english")`.

In this project, each page becomes a sparse vector of TF-IDF term weights. A
query is converted to a vector in the same feature space, so its terms can be
compared directly with every page.

### Why cosine similarity?

Cosine similarity measures the angle (directional similarity) between two
vectors. It is useful for text because it focuses on how their weighted terms
line up rather than simply comparing page lengths. Scikit-learn returns scores
from 0 to 1 for these non-negative TF-IDF vectors; a larger score means a
closer match.

## Installation

From the project root, create and activate a virtual environment, then install
the listed dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv/Scripts/Activate.ps1` instead.

The runtime requirements are listed in `requirements.txt`.

## Usage

Place your selectable-text PDF at `data/document.pdf` locally, or provide any
other PDF path. PDFs under `data/` are intentionally ignored by Git:

```bash
python search.py data/document.pdf "neural networks"
```

Optional top-k ranking:

```bash
python search.py data/document.pdf "neural networks" --top-k 5
```

`--top-k` defaults to 1, so the basic two-argument command remains fully
supported. A query containing only whitespace is rejected. If none of its
terms occur in the fitted TF-IDF vocabulary (including when all terms are
English stop words), the program explains that there are no meaningful
matching terms.

To inspect extracted page text directly:

```bash
python -c 'from src.pdf_reader import extract_pages; pages = extract_pages("data/document.pdf"); print("\n".join("Page {}: {} characters".format(p["page_number"], len(p["text"])) for p in pages[:5]))'
```

## Example output

For the local PDF used during development, the command
`python search.py data/document.pdf "neural networks"` returned:

```text
========================================
       PDF PAGE SIMILARITY SEARCH
========================================

PDF:
data/document.pdf

Query:
neural networks

Most Relevant Page:
264

Similarity Score:
0.5740

Relevant Text:
----------------------------------------
neural networks backward pass, Neural Networks: The Backward Pass-The overall
loss gradient building from scratch, Neural Networks from Scratch-Code, Deep
Learning from Scratch-NeuralNetwork: Code, Our First Deep Learning Model ...
----------------------------------------
```

For long pages, the displayed text is limited to the first 1,200 characters and
is labeled as an excerpt. The complete cleaned page text is still used for
similarity calculation.

## Project structure

```text
.
├── data/
│   ├── .gitkeep              # keeps the local input directory in Git
│   └── document.pdf          # local PDF; intentionally not committed
├── src/
│   ├── __init__.py
│   ├── pdf_reader.py         # one-based page text extraction and PDF errors
│   ├── text_processor.py     # whitespace normalization
│   └── search_engine.py      # reusable TF-IDF index and similarity ranking
├── tests/
│   ├── __init__.py
│   ├── test_pdf_reader.py    # generated-PDF extraction tests
│   ├── test_text_processor.py
│   └── test_search_engine.py # small artificial page collections
├── search.py                # argparse CLI and excerpt display
├── requirements.txt
├── README.md
└── .gitignore
```

## Running tests

The tests use Python's built-in `unittest` and generate small PDFs/pages, so the
book is not needed to run the unit suite:

```bash
python -m unittest discover -v
```

## Limitations

- TF-IDF is keyword-based; it does not fully understand semantic meaning or
  synonyms.
- PDF extraction can have reading-order and formatting issues.
- Tables and multi-column layouts may not extract perfectly.
- Treating one page as one vector can be coarse when only a small part of a
  page is relevant.
- Image-only pages need OCR; this application does not perform OCR.
- A query with no terms in the fitted vocabulary cannot produce a meaningful
  match.

## Future improvements

Possible extensions (not implemented here) include sentence-transformer
embeddings and semantic search, BM25 ranking, FAISS for approximate vector
search at larger scale, chunk-level retrieval, OCR, and indexing multiple PDFs.

## Viva Questions

1. **What is TF-IDF?** — A term-weighting method that combines how often a
   term occurs in a document with how rare it is across the document collection.
2. **Why is TF-IDF used?** — It emphasizes terms that help distinguish a page
   from the other pages, rather than giving every word equal importance.
3. **What is cosine similarity?** — A measure of the cosine of the angle
   between two vectors; here it compares a query vector with page vectors.
4. **Why cosine similarity instead of Euclidean distance?** — Cosine focuses on
   vector direction and is less affected by vector magnitude/page length, which
   is useful for sparse text vectors.
5. **What is a vectorizer?** — A component that learns a vocabulary and
   converts text into numeric feature vectors; this project uses
   `TfidfVectorizer`.
6. **What are stop words?** — Very common words such as “the” and “is” that
   usually contribute little to keyword relevance; scikit-learn removes its
   English stop-word list here.
7. **Why process pages separately?** — So the result can identify and return
   the original PDF page most relevant to the query.
8. **Why preserve page numbers?** — Blank pages are excluded from indexing, but
   users need page numbers that still match the original PDF.
9. **How is the query converted into a vector?** — The already-fitted TF-IDF
   vectorizer transforms it using the page collection's learned vocabulary and
   IDF weights.
10. **How do we find the most relevant page?** — Calculate cosine similarity
    between the query vector and each page vector, then choose the largest
    score (or sort scores for top-k results).
11. **What happens if a query word does not occur in the PDF?** — That word is
    outside the learned vocabulary and contributes no feature. If no query terms
    are in the vocabulary, the program reports no meaningful matching terms.
12. **What are the limitations of TF-IDF?** — It relies on shared words, does
    not understand context/synonyms, and may be coarse when a whole page is
    represented by one vector.
13. **What is semantic search?** — Search that compares meaning and context,
    often by comparing learned dense vector representations rather than exact
    word-weight overlap.
14. **How would embeddings improve this project?** — Embeddings can place
    semantically related text near one another even when the exact words differ;
    they add model/runtime complexity and are intentionally outside this
    classical baseline.
15. **How could this system be scaled to thousands of PDFs?** — Extract and
    index documents incrementally, persist metadata and vectors, use an
    inverted index or an approximate nearest-neighbor index such as FAISS, and
    consider BM25 or chunk-level retrieval depending on the search needs.
