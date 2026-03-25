# preprocessing script for iitj corpus
# loads raw scraped text, cleans it, tokenizes, and saves corpus.txt
# also computes vocab stats and generates word cloud if possible


import os
import re
import json
import hashlib
import logging
from collections import Counter

import nltk
from nltk.corpus import stopwords

# spacy is the main tokenizer/lemmatizer now - much better than porter stemmer
# porter was turning "university" -> "univers" which is just wrong
# install: pip install spacy && python -m spacy download en_core_web_sm
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", disable=["parser"])
    _nlp.add_pipe("sentencizer")   # fast sentence splitting without full dep parse
    USE_SPACY = True
except (ImportError, OSError):
    print("[warn] spacy not found, falling back to nltk + wordnet lemmatizer")
    print("       to fix: pip install spacy && python -m spacy download en_core_web_sm")
    USE_SPACY = False

# gensim Phrases learns bigrams/trigrams from the actual corpus data
# way better than hardcoding BIGRAMS by hand - finds collocations we'd miss
# install: pip install gensim
try:
    from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
    USE_PHR = True
except ImportError:
    print("[warn] gensim not found, phrase detection disabled")
    print("       to fix: pip install gensim")
    USE_PHR = False

# minhash for near-duplicate document detection
# catches cases where same page got scraped twice with minor differences
# install: pip install datasketch
try:
    from datasketch import MinHash, MinHashLSH
    USE_MH = True
except ImportError:
    print("[warn] datasketch not found, near-dup doc detection disabled")
    print("       to fix: pip install datasketch")
    USE_MH = False

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)   # needed for fallback lemmatizer

# ── paths ──────────────────────────────────────────────────────────────────────
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

ALL_SW = S_W | D_SW

# only keep content-bearing POS tags - everything else is grammatical filler
# NOUN=things, PROPN=proper names, VERB=actions, ADJ=descriptors, ADV=modifiers
# way more principled than a stopword list - spacy decides, not us
KEEP_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

# min frequency threshold - 2 so rare typos and scraping artifacts die
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

# hardcoded bigrams still useful as a pre-tokenization pass
# gensim Phrases will catch more data-driven ones later
# but these are 100% certain so we join them before spacy sees the text
# this way spacy treats "machine_learning" as ONE token, not two
BIGRAMS = [
    (r"\bcomputer\s+science\b",          "computer_science"),
    (r"\belectrical\s+engineering\b",    "electrical_engineering"),
    (r"\bmechanical\s+engineering\b",    "mechanical_engineering"),
    (r"\bcivil\s+engineering\b",         "civil_engineering"),
    (r"\bchemical\s+engineering\b",      "chemical_engineering"),
    (r"\bmaterials?\s+engineering\b",    "materials_engineering"),
    (r"\bdata\s+science\b",              "data_science"),
    (r"\bmachine\s+learning\b",          "machine_learning"),
    (r"\bdeep\s+learning\b",             "deep_learning"),
    (r"\bnatural\s+language\b",          "natural_language"),
    (r"\bartificial\s+intelligence\b",   "artificial_intelligence"),
    (r"\bresearch\s+scholar\b",          "research_scholar"),
    (r"\bfaculty\s+member\b",            "faculty_member"),
    (r"\bacademic\s+year\b",             "academic_year"),
    (r"\bboard\s+of\s+governors\b",      "board_of_governors"),
    (r"\bphd\s+student\b",               "phd_student"),
    (r"\bphd\s+scholar\b",               "phd_scholar"),
    (r"\bphd\s+thesis\b",                "phd_thesis"),
    (r"\bgraduate\s+student\b",          "graduate_student"),
    (r"\bundergraduate\s+student\b",     "undergraduate_student"),
    (r"\badmission\s+test\b",            "admission_test"),
    (r"\bcut.?off\b",                    "cutoff"),
    (r"\bword2vec\b",                    "word2vec"),
]

# precompile all bigram patterns once at module load - cheaper than recompiling every doc
_BIGRAM_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in BIGRAMS]

# track seen sentence hashes for dedup - module-level so it persists across all docs
_seen_s_hashes: set = set()


def ld_raw():
    # load all raw txt files from raw_data directory
    documents = []
    if not os.path.exists(R_DIR):
        log.error("raw data directory not found: %s", R_DIR)
        log.error("run scraper.py first!")
        return documents

    for fname in sorted(os.listdir(R_DIR)):
        if fname.endswith('.txt'):
            filepath = os.path.join(R_DIR, fname)
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            if text.strip():
                documents.append({"filename": fname, "text": text})

    log.info("loaded %d raw documents.", len(documents))
    return documents


def dedup_docs(docs, threshold=0.85):
    # remove near-duplicate documents using minhash LSH
    # catches cases where same iitj page got scraped from two different urls
    # threshold=0.85 means 85% token overlap = duplicate - tune down if too aggressive
    if not USE_MH:
        return docs

    lsh  = MinHashLSH(threshold=threshold, num_perm=128)
    uniq = []

    for i, doc in enumerate(docs):
        toks = set(doc["text"].lower().split())
        m = MinHash(num_perm=128)
        for t in toks:
            m.update(t.encode())
        try:
            if lsh.query(m):   # near-duplicate already in index - skip
                log.debug("dropping near-dup: %s", doc["filename"])
                continue
            lsh.insert(str(i), m)
            uniq.append(doc)
        except Exception:
            uniq.append(doc)   # insert collision edge case - just keep it

    removed = len(docs) - len(uniq)
    if removed:
        log.info("near-dup removal: dropped %d / %d docs", removed, len(docs))
    return uniq


def strip_boilerplate(text):
    # remove recurring header/footer text from iitj website
    # this stuff appears on literally every page and means nothing
    for pat in BOILERPLATE:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE | re.DOTALL)
    return text


def cln_d(text):
    # strip footers/navbars first before anything else
    text = strip_boilerplate(text)

    # normalize academic abbreviations early so bigrams can match them correctly
    text = re.sub(r'\bb\.?\s*tech(?:\s*/\s*bs)?\b', ' btech ',       text, flags=re.IGNORECASE)
    text = re.sub(r'\bm\.?\s*tech\b',               ' mtech ',       text, flags=re.IGNORECASE)
    text = re.sub(r'\bm\.?\s*sc\b',                 ' msc ',         text, flags=re.IGNORECASE)
    text = re.sub(r'\bph\.?\s*d\b',                 ' phd ',         text, flags=re.IGNORECASE)
    text = re.sub(r'\bug\b',                         ' undergraduate ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpg\b',                         ' postgraduate ', text, flags=re.IGNORECASE)

    # consolidate all exam/evaluation-related terms into a single token
    # so the model sees them as the same concept regardless of how they're written
    text = re.sub(
        r'\b(?:examinations?|quizzes?|mid-?sem(?:ester)?|end-?sem(?:ester)?|viva(?:-voce)?|assessments?|evaluations?|interviews?)\b',
        ' exam ', text, flags=re.IGNORECASE
    )

    # join known academic phrases into single tokens BEFORE tokenization
    # doing it here means spacy sees "machine_learning" as one unit, not two
    for pat, repl in _BIGRAM_RE:
        text = pat.sub(f' {repl} ', text)

    # handle slash-separated terms like mtech/phd -> split them
    text = re.sub(r'(\w+)/(\w+)', r'\1 \2', text)

    # standard cleanup
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)        # non-ascii (hindi etc)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # urls
    text = re.sub(r'\S+@\S+\.\S+', '', text)           # emails
    text = re.sub(r'\+?\d[\d\s\-]{8,}\d', '', text)    # phone numbers
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)           # html entities
    text = re.sub(r'[|\*\#\>\<\{\}\[\]\\~`^]', ' ', text)  # special chars
    text = re.sub(r'\b\d+\b', '', text)                # standalone numbers
    text = re.sub(r'\.{2,}', '.', text)                # excessive dots
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _dedup_s(sent):
    # returns True if we should keep this sentence (haven't seen it before)
    # md5 of lowercased+stripped text - catches exact boilerplate repeats across docs
    key = re.sub(r'\s+', ' ', sent.lower().strip())
    h   = hashlib.md5(key.encode()).hexdigest()
    if h in _seen_s_hashes:
        return False
    _seen_s_hashes.add(h)
    return True


def tok_c(text):
    # tokenize and lemmatize text - returns List[List[str]]
    # IMPORTANT: must return list of sentences, not a flat list!
    # flat list kills word2vec context windows completely
    if USE_SPACY:
        return _tok_spacy(text)
    return _tok_nltk(text)


def _tok_spacy(text):
    # spacy path - lemmatization + POS filtering
    # lemmatizer uses a dictionary so "universities" -> "university" (not "univers")
    # POS filter keeps only content words - this is the main upgrade over v1
    doc    = _nlp(text)
    result = []

    for sent in doc.sents:
        if len(sent) < 4:         # skip nav fragments like "Home > Academics"
            continue
        if not _dedup_s(sent.text):  # skip repeated boilerplate sentences
            continue

        cleaned = []
        for tok in sent:
            # hardcoded bigram tokens already have underscores - keep as-is
            if '_' in tok.text and not tok.is_space:
                t = tok.text.lower()
                if t not in ALL_SW and len(t) > 2:
                    cleaned.append(t)
                continue

            # POS filter - only keep content-bearing words
            if tok.pos_ not in KEEP_POS:
                continue
            if tok.is_stop or tok.is_punct or tok.is_space:
                continue

            lemma = tok.lemma_.lower().strip()

            if not lemma or len(lemma) < 2:
                continue
            if re.match(r'^[\W\d]+$', lemma):   # pure punct/numbers post-lemma
                continue
            if lemma in ALL_SW:
                continue
            if len(lemma) > 30:   # scraping artifact
                continue

            cleaned.append(lemma)

        if len(cleaned) >= 3:
            result.append(cleaned)

    return result


def _tok_nltk(text):
    # fallback when spacy isn't available
    # wordnet lemmatizer is worse but still better than porter stemmer
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.stem import WordNetLemmatizer
    wnl    = WordNetLemmatizer()
    result = []

    for sent in sent_tokenize(text):
        if not _dedup_s(sent):
            continue
        cleaned = []
        for tok in word_tokenize(sent.lower()):
            if re.match(r'^[^\w_]+$', tok):
                continue
            if len(tok) == 1 and tok not in ('a', 'i'):
                continue
            if re.match(r'^\d+$', tok):
                continue
            if tok in ALL_SW or len(tok) > 30:
                continue
            lemma = wnl.lemmatize(tok)   # defaults to noun pos - good enough for fallback
            if len(lemma) >= 2 and lemma not in ALL_SW:
                cleaned.append(lemma)

        if len(cleaned) >= 3:
            result.append(cleaned)

    return result


def _run_phrases(all_sents):
    # learn bigrams then trigrams from the actual corpus using gensim Phrases
    # this catches collocations we'd never think to hardcode manually
    # e.g. "senate_academic_affairs", "sponsored_research_project"
    # min_count=3 and threshold=8 are conservative for a small academic corpus
    # increase min_count if corpus is large (>1M tokens)
    if not USE_PHR:
        return all_sents

    log.info("training bigram phrase model...")
    bg = Phrases(all_sents, min_count=3, threshold=8,
                 connector_words=ENGLISH_CONNECTOR_WORDS)

    log.info("training trigram phrase model...")
    tg = Phrases(bg[all_sents], min_count=3, threshold=8,
                 connector_words=ENGLISH_CONNECTOR_WORDS)

    phrased = [tg[bg[s]] for s in all_sents]
    log.info("phrase detection done.")
    return phrased


def b_corp(documents, min_freq=M_FREQ):
    # build full corpus from all documents
    all_sents = []
    meta      = []    # provenance info for jsonl output
    t_cnt     = Counter()

    for doc in documents:
        cln_text = cln_d(doc["text"])
        sentences = tok_c(cln_text)   # List[List[str]]
        for s_i, sent in enumerate(sentences):
            all_sents.append(sent)
            t_cnt.update(sent)
            meta.append({"doc": doc["filename"], "sent_id": s_i})

    # run gensim Phrases to detect data-driven bigrams/trigrams
    # updates all_sents with joined tokens like "machine_learning"
    all_sents = _run_phrases(all_sents)

    # recount after phrase detection since token shapes changed
    t_cnt = Counter()
    for sent in all_sents:
        t_cnt.update(sent)

    # filter out rare words - typos and scraping noise usually appear once
    voc = {word for word, count in t_cnt.items() if count >= min_freq}

    # filter sentences to only vocab words
    # also drop sentences that become too short after filtering
    filt_sents = []
    filt_meta  = []
    for sent, m in zip(all_sents, meta):
        filtered = [w for w in sent if w in voc]
        if len(filtered) >= 3:
            m["tokens"] = filtered
            filt_sents.append(filtered)
            filt_meta.append(m)

    log.info("total sentences (after filtering): %d", len(filt_sents))
    return filt_sents, t_cnt, voc, filt_meta


def sv_corp(sentences, filepath):
    # save corpus as one sentence per line, space-separated tokens
    # this is the exact format word2vec / fasttext expect
    with open(filepath, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(' '.join(sent) + '\n')
    log.info("corpus (txt)   -> %s", filepath)


def sv_corp_jsonl(meta, filepath):
    # save corpus as jsonl with provenance info - useful for bert fine-tuning later
    # each line: {"doc": "filename.txt", "sent_id": 0, "tokens": [...]}
    with open(filepath, 'w', encoding='utf-8') as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    log.info("corpus (jsonl) -> %s", filepath)


def cmp_st(sentences, t_cnt, voc, num_docs):
    # compute and print corpus statistics
    t_toks  = sum(len(s) for s in sentences)
    t_sents = len(sentences)

    # hapax legomena = words that appear exactly once - high count means noisy corpus
    hapax   = sum(1 for c in t_cnt.values() if c == 1)
    # how many vocab types are multi-word phrases (have underscore)
    phrases = sum(1 for w in voc if '_' in w)
    # type-token ratio - lower = more repetitive (expected for a domain corpus)
    ttr     = round(len(voc) / max(t_toks, 1), 4)

    stats = {
        "num_documents":       num_docs,
        "num_sentences":       t_sents,
        "total_tokens":        t_toks,
        "vocabulary_size":     len(voc),
        "hapax_legomena":      hapax,
        "phrase_types":        phrases,
        "avg_sentence_length": round(t_toks / max(t_sents, 1), 2),
        "type_token_ratio":    ttr,
        "top_50_words":        t_cnt.most_common(50),
        "top_20_phrases":      [(w, c) for w, c in t_cnt.most_common(200) if '_' in w][:20],
    }

    log.info("=" * 60)
    log.info("total documents:       %d",   stats['num_documents'])
    log.info("total sentences:       %d",   stats['num_sentences'])
    log.info("total tokens:          %d",   stats['total_tokens'])
    log.info("vocabulary size:       %d  (hapax: %d | phrases: %d)", len(voc), hapax, phrases)
    log.info("type-token ratio:      %.4f  (lower = more repetitive)", ttr)
    log.info("avg sentence length:   %.2f", stats['avg_sentence_length'])
    log.info("")
    log.info("top 20 most frequent words:")
    for word, count in stats['top_50_words'][:20]:
        log.info("  %-25s %6d", word, count)
    log.info("")
    log.info("top 10 detected phrases:")
    for phrase, count in stats['top_20_phrases'][:10]:
        log.info("  %-30s %6d", phrase, count)
    log.info("=" * 60)

    stats_file = os.path.join(O_DIR, "corpus_statistics.json")
    stats_s    = stats.copy()
    stats_s["top_50_words"]   = [[w, c] for w, c in stats["top_50_words"]]
    stats_s["top_20_phrases"] = [[w, c] for w, c in stats["top_20_phrases"]]
    with open(stats_file, 'w') as f:
        json.dump(stats_s, f, indent=2)
    log.info("stats saved to: %s", stats_file)

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
            collocations=False,    # we already handled collocations via Phrases above
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
        log.info("word cloud saved to: %s", wc_path)
    except ImportError:
        log.warning("[skip] wordcloud not installed, skipping word cloud generation")


def main():
    # load raw docs
    documents = ld_raw()
    if not documents:
        return

    # drop near-duplicate documents before anything else
    documents = dedup_docs(documents, threshold=0.85)

    # build corpus: clean -> tokenize/lemmatize -> phrase detection -> vocab filter
    sentences, t_cnt, voc, meta = b_corp(documents, min_freq=M_FREQ)

    if not sentences:
        log.error("empty corpus after preprocessing - check raw_data/ directory")
        return

    # save in both formats
    corpus_path = os.path.join(os.path.dirname(__file__), "corpus.txt")
    sv_corp(sentences, corpus_path)

    jsonl_path = os.path.join(os.path.dirname(__file__), "corpus.jsonl")
    sv_corp_jsonl(meta, jsonl_path)

    # stats + word cloud
    cmp_st(sentences, t_cnt, voc, len(documents))
    gen_wc(t_cnt)

    log.info("preprocessing complete!")


if __name__ == "__main__":
    main()
