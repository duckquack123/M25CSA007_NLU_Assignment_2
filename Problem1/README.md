# Problem 1: Word2Vec on IIT Jodhpur Corpus

This project involves curating a domain-specific corpus from IIT Jodhpur's web resources and training Word2Vec models (CBOW and Skip-gram) from scratch using PyTorch and Gensim.

## 🚀 Pipeline Overview

### 1. Web Scraping (`scraper.py`)
- **BFS Crawler**: Navigates the IIT Jodhpur website to a depth of 5.
- **Resiliency**: Handles timeouts and retries for slow servers.
- **Content**: Extracts text from HTML and PDF (academic regulations).
- **Output**: Raw text files in `raw_data/`.

### 2. Preprocessing (`preprocess.py`)
- **Cleaning**: Removes boilerplate, HTML tags, and non-ASCII characters.
- **NLP Pipeline**: Uses **SpaCy** for high-quality sentence splitting and lemmatization.
- **Normalization**: Consolidates academic terms (e.g., *viva* -> *exam*).
- **Filtering**: Removes low-frequency tokens and non-content words.
- **Output**: `corpus.txt` and `outputs/corpus_statistics.json`.

### 3. Model Training (`word2vec_scratch.py`)
- **Implementation**: Custom PyTorch modules for CBOW and Skip-gram with Negative Sampling.
- **Experiments**: Systematic hyperparameter search across:
  - Embedding dimensions: 50, 100, 200.
  - Window sizes: 2, 3, 5.
  - Negative samples: 5, 10, 15.
- **Output**: Trained weights in `outputs/models/` and summary plots in `outputs/training_curves.png`.

### 4. Semantic Analysis & Visualization
- **`analysis.py`**: Performs nearest-neighbor lookups and analogy tests (e.g., *UG:BTech :: PG:?*).
- **`visualize.py`**: Generates PCA and t-SNE projections to visualize semantic clusters.
- **`word2vec_gensim.py`**: Provides a baseline comparison using the professional Gensim library.

## 🛠️ Setup & Usage

1. **Install Dependencies**:
   ```bash
   pip install spacy nltk gensim datasketch wordcloud matplotlib scikit-learn
   python -m spacy download en_core_web_sm
   ```

2. **Run Pipeline**:
   ```bash
   python scraper.py        # Collect data
   python preprocess.py     # Clean & tokenize
   python word2vec_scratch.py  # Train models
   python analysis.py       # semantic tests
   python visualize.py      # Generate plots
   ```

## 📊 Key Results
- **Vocabulary Size**: ~6,990 tokens.
- **Top Concepts**: *student*, *research*, *program*, *course*, *academic*.
- **Training Curves**: Saved in `outputs/training_curves.png`.
