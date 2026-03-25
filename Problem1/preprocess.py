# preprocessing script for iitj corpus
# loads raw scraped text, cleans it, tokenizes, and saves corpus.txt
# also computes vocab stats and generates word cloud if possible

import os
import re
import json
import nltk
from collections import Counter

# make sure nltk data is present
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# using porter stemmer to reduce vocab size
ps = PorterStemmer()

R_DIR = os.path.join(os.path.dirname(__file__), "raw_data")
O_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(O_DIR, exist_ok=True)

# standard english stopwords
S_W = set(stopwords.words('english'))

# adding domain-specific stopwords - these appear everywhere on iitj site
# but carry almost no semantic meaning for word2vec
# they just dilute the context window
D_SW = {
    "iit", "jodhpur", "iitj", "institute", "technology", "indian",
    "campus", "portal", "website", "page", "copyright", "reserved",
    "reach", "manager", "committee", "updated", "last", "click",
    "please", "email", "developed", "designed", "automation",
    "infrastructure", "digital", "nagaur", "road", "karwar",
    "rajasthan", "india", "nh", "pm", "am", "jan", "feb", "mar",
    "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "repository", "web", "information", "internal", "http", "www",
    "also", "iitjodhpur", "n.h",
}

# final stopword set = english + domain-specific
ALL_SW = S_W | D_SW

# min frequency threshold - increased to 2 so rare typos and scraping artifacts die
M_FREQ = 2

# boilerplate text that appears on every page - we strip this before processing
# idk why iitj puts the same nav links on every single page
BOILERPLATE = [
    r"how to reach iitj.*?institute repository.*?last updated.*?(?:am|pm)",
    r"copyright.*?all rights reserved.*?developed by.*?nagaur road",
    r"web information manager.*?internal committee",
    r"for any comments/enquiries/feedback.*?email",
    r"this portal is owned.*?automation",
]

# common academic multi-word phrases that should be treated as single concepts
# word2vec needs bigrams like "computer_science" to learn the right semantics
# without this, "computer" and "science" get separate unrelated embeddings
BIGRAMS = [
    (r"\bcomputer\s+science\b", "computer_science"),
    (r"\belectrical\s+engineering\b", "electrical_engineering"),
    (r"\bmechanical\s+engineering\b", "mechanical_engineering"),
    (r"\bcivil\s+engineering\b", "civil_engineering"),
    (r"\bchemical\s+engineering\b", "chemical_engineering"),
    (r"\bmaterials?\s+engineering\b", "materials_engineering"),
    (r"\bdata\s+science\b", "data_science"),
    (r"\bmachine\s+learning\b", "machine_learning"),
    (r"\bdeep\s+learning\b", "deep_learning"),
    (r"\bnatural\s+language\b", "natural_language"),
    (r"\bartificial\s+intelligence\b", "artificial_intelligence"),
    (r"\bresearch\s+scholar\b", "research_scholar"),
    (r"\bfaculty\s+member\b", "faculty_member"),
    (r"\bacademic\s+year\b", "academic_year"),
    (r"\bboard\s+of\s+governors\b", "board_of_governors"),
    (r"\bphd\s+student\b", "phd_student"),
    (r"\bphd\s+scholar\b", "phd_scholar"),
    (r"\bphd\s+thesis\b", "phd_thesis"),
    (r"\bgraduate\s+student\b", "graduate_student"),
    (r"\bundergraduate\s+student\b", "undergraduate_student"),
    (r"\badmission\s+test\b", "admission_test"),
    (r"\bcut.?off\b", "cutoff"),
    (r"\bword2vec\b", "word2vec"),
]


def ld_raw():
    # load all raw txt files from raw_data directory
    documents = []
    if not os.path.exists(R_DIR):
        print(f"raw data directory not found: {R_DIR}")
        print("run scraper.py first!")
        return documents

    for fname in sorted(os.listdir(R_DIR)):
        if fname.endswith('.txt'):
            filepath = os.path.join(R_DIR, fname)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            if text.strip():
                documents.append({"filename": fname, "text": text})

    print(f"loaded {len(documents)} raw documents.")
    return documents


def strip_boilerplate(text):
    # remove recurring header/footer text from iitj website
    # this stuff appears on literally every page and means nothing
    for pat in BOILERPLATE:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE | re.DOTALL)
    return text


def cln_d(text):
    # strip footers/navbars first before anything else
    text = strip_boilerplate(text)

    # normalize academic abbreviations early so bigrams can match them
    text = re.sub(r'\bb\.?\s*tech(?:\s*/\s*bs)?\b', ' btech ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bm\.?\s*tech\b', ' mtech ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bm\.?\s*sc\b', ' msc ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bph\.?\s*d\b', ' phd ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bug\b', ' undergraduate ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpg\b', ' postgraduate ', text, flags=re.IGNORECASE)

    # consolidate all exam/evaluation-related terms into a single token
    # so the model sees them as the same concept
    text = re.sub(
        r'\b(?:examinations?|quizzes?|mid-?sem(?:ester)?|end-?sem(?:ester)?|viva(?:-voce)?|assessments?|evaluations?|interviews?)\b',
        ' exam ', text, flags=re.IGNORECASE
    )

    # join common academic phrases into single tokens before tokenization
    # this is important for the model to learn domain-specific bigrams
    for pat, repl in BIGRAMS:
        text = re.sub(pat, f' {repl} ', text, flags=re.IGNORECASE)

    # handle slash-separated terms like mtech/phd -> split them
    # usually these are either-or program types
    text = re.sub(r'(\w+)/(\w+)', r'\1 \2', text)

    # cleanup
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # remove non-ascii
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove urls
    text = re.sub(r'\S+@\S+\.\S+', '', text)      # remove emails
    text = re.sub(r'\+?\d[\d\s\-]{8,}\d', '', text)  # remove phone nums
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)      # html entities
    text = re.sub(r'[|\*\#\>\<\{\}\[\]\\~`^]', ' ', text)  # special chars
    text = re.sub(r'\b\d+\b', '', text)            # standalone numbers
    text = re.sub(r'\.{2,}', '.', text)            # excessive dots
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tok_c(text):
    # sentence tokenize first so word2vec sees proper context windows
    # IMPORTANT: return List[List[str]] not a flat list!
    # if we return flat list, b_corp treats each word as a separate sentence
    # which kills the context window and makes word2vec useless
    sentences = sent_tokenize(text)
    result = []

    for sent in sentences:
        tokens = word_tokenize(sent.lower())
        cleaned = []
        for tok in tokens:
            # skip pure punctuation
            if re.match(r'^[^\w_]+$', tok):
                continue
            # skip single chars (except a, i)
            if len(tok) == 1 and tok not in ('a', 'i'):
                continue
            # skip pure numbers
            if re.match(r'^\d+$', tok):
                continue
            # skip domain + english stopwords
            if tok in ALL_SW:
                continue
            # skip very long tokens (scraping artifacts usually)
            if len(tok) > 30:
                continue
            # stem the token
            stemmed = ps.stem(tok)
            # skip if stem is too short (< 2 chars) - usually not useful
            if len(stemmed) < 2:
                continue
            cleaned.append(stemmed)

        # only keep sentences with at least 3 tokens
        # single-word "sentences" don't provide context for word2vec
        if len(cleaned) >= 3:
            result.append(cleaned)

    return result


def b_corp(documents, min_freq=M_FREQ):
    # build full corpus from all documents
    all_sentences = []
    t_cnt = Counter()

    for doc in documents:
        cleaned_text = cln_d(doc["text"])
        # tok_c returns List[List[str]] now - each elem is a sentence
        sentences = tok_c(cleaned_text)
        for sent in sentences:
            all_sentences.append(sent)
            t_cnt.update(sent)  # count word frequencies

    # filter out rare words (likely typos or scraping noise)
    voc = {word for word, count in t_cnt.items() if count >= min_freq}

    # filter sentences to only vocab words
    # also drop sentences that become too short after filtering
    filtered_sentences = []
    for sent in all_sentences:
        filtered = [w for w in sent if w in voc]
        if len(filtered) >= 3:
            filtered_sentences.append(filtered)

    print(f"total sentences (after filtering): {len(filtered_sentences)}")
    return filtered_sentences, t_cnt, voc


def sv_corp(sentences, filepath):
    # save corpus as one sentence per line, space-separated tokens
    with open(filepath, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(' '.join(sent) + '\n')
    print(f"corpus saved to: {filepath}")


def cmp_st(sentences, t_cnt, voc, num_docs):
    # compute and print corpus statistics
    t_toks = sum(len(s) for s in sentences)
    t_sents = len(sentences)

    stats = {
        "num_documents": num_docs,
        "num_sentences": t_sents,
        "total_tokens": t_toks,
        "vocabulary_size": len(voc),
        "avg_sentence_length": round(t_toks / max(t_sents, 1), 2),
        "top_50_words": t_cnt.most_common(50),
    }

    print(f"total documents:       {stats['num_documents']}")
    print(f"total sentences:       {stats['num_sentences']}")
    print(f"total tokens:          {stats['total_tokens']:,}")
    print(f"vocabulary size:       {stats['vocabulary_size']:,}")
    print(f"avg sentence length:   {stats['avg_sentence_length']}")
    print(f"\ntop 20 most frequent words:")
    for word, count in stats['top_50_words'][:20]:
        print(f"  {word:20s} {count:6d}")
    print(f"{'='*60}")

    stats_file = os.path.join(O_DIR, "corpus_statistics.json")
    stats_serializable = stats.copy()
    stats_serializable["top_50_words"] = [[w, c] for w, c in stats["top_50_words"]]
    with open(stats_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"stats saved to: {stats_file}")

    return stats


def gen_wc(t_cnt):
    # generate word cloud - optional, skips if wordcloud not installed
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        freq_dict = {word: count for word, count in t_cnt.items()
                     if word not in ALL_SW and len(word) > 2}

        wc = WordCloud(
            width=1200, height=600,
            background_color='white',
            max_words=150,
            colormap='viridis',
            contour_width=2,
            contour_color='steelblue'
        ).generate_from_frequencies(freq_dict)

        plt.figure(figsize=(15, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('word cloud - iit jodhpur corpus', fontsize=20, fontweight='bold')
        plt.tight_layout()

        wc_path = os.path.join(O_DIR, "wordcloud.png")
        plt.savefig(wc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"word cloud saved to: {wc_path}")
    except ImportError:
        print("[skip] wordcloud not installed, skipping word cloud generation")


def main():
    documents = ld_raw()
    if not documents:
        return

    # build and filter corpus
    sentences, t_cnt, voc = b_corp(documents, min_freq=M_FREQ)

    corpus_path = os.path.join(os.path.dirname(__file__), "corpus.txt")
    sv_corp(sentences, corpus_path)

    cmp_st(sentences, t_cnt, voc, len(documents))
    gen_wc(t_cnt)

    print("\npreprocessing complete!")


if __name__ == "__main__":
    main()
