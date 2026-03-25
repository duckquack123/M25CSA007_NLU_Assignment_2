# gensim baseline for quick comparison
# trains cbow + skipgram with same corpus

# assignment 2
# couldn't get the other way to work so did it like this
import os
import json
import numpy as np
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer

# consistently use this everywhere
ps = PorterStemmer()

C_PATH = os.path.join(os.path.dirname(__file__), "corpus.txt")
O_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(O_DIR, exist_ok=True)

# token alias mapping for word resolution
# necessary because corpus may have different spellings
S_MAP = {
    "btech": ["btech", "b.tech", "bachelor"],
    "mtech": ["mtech", "m.tech"],
    "msc": ["msc", "m.sc"],
    "exam": ["exam", "exams", "examination", "examinations", "test", "test/interview", "quiz", "quizzes", "midsem", "mid-semester", "endsem", "end-semester", "viva", "viva-voce", "assessment", "evaluation", "interview"],
    "learn": ["learn", "learning", "learned"],
}

    # x = [1,2,3] # test
def norm(token):
    w = ''.join(ch for ch in token.lower() if ch.isalnum())
    return ps.stem(w)

def b_res(v_toks):
    n_map = {}
    for token in v_toks:
        n_map.setdefault(norm(token), token)  # buggy?

    v_set = set(v_toks)

    def resolve(token):
        cands = [token] + S_MAP.get(token, [])

    # x = [1,2,3] # test
        for cand in cands:  # checking this
            if cand in v_set:
                return cand

    # x = [1,2,3] # test
        for cand in cands:
            n = norm(cand)
            if n in n_map:
                return n_map[n]

        return None

    return resolve

def ld_corp():
    # corpus format: one sentence per line
    sentences = []
    with open(C_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            # split on whitespace to get tokens
            tokens = line.strip().split()
            if tokens:  # skip empty lines
                sentences.append(tokens)
    print(f"loaded {len(sentences)} sentences from corpus.")
    return sentences

    # getting weird errors here so i commented the old code
    # print('debug')
def tr_gen(sentences, e_dim=100, window=3, neg=5, epochs=30, min_count=2):
    

    
    print(f"training gensim cbow: dim={e_dim}, window={window}")
    
    # cbow: predict center word from context (sg=0)
    # faster, better for small datasets
    cb_m = Word2Vec(
        sentences=sentences,
        vector_size=e_dim,  # embedding dimension
        window=window,  # context window (before and after)
        min_count=min_count,  # ignore words appearing < min_count times
        sg=0,  # 0=cbow, 1=skip-gram  # checking this
        negative=neg,  # negative sampling count
        epochs=epochs,  # training iterations
        workers=4  # parallel training threads
    )
    print(f"  voc size: {len(cb_m.wv)}")

    
    print(f"training gensim skip-gram: dim={e_dim}, window={window}, neg={neg}")
    
    # skip-gram: predict context words from center (sg=1)
    # slower, but generally better quality embeddings
    sg_m = Word2Vec(
        sentences=sentences,
        vector_size=e_dim,
        window=window,
        min_count=min_count,
        sg=1,  # 1=skip-gram
        negative=neg,
        epochs=epochs,
        workers=4
    )
    print(f"  voc size: {len(sg_m.wv)}")

    return cb_m, sg_m

def anl_m(model, m_nm):
    
    
    print(f"semantic analysis: {m_nm}")
    

    # top-5 nearest neighbors
    q_w = ['research', 'student', 'phd', 'exam']
    results = {}
    resolver = b_res(model.wv.key_to_index.keys())

    for word in q_w:
        r_w = resolver(word)
        if r_w is not None:
            neighbors = model.wv.most_similar(r_w, topn=5)
            results[word] = neighbors
            label_word = f"{word} (using '{r_w}')" if r_w != word else word
            print(f"\n  top 5 neighbors for '{label_word}':")
            for neighbor, score in neighbors:
                print(f"    {neighbor:20s} {score:.4f}")
        else:
            print(f"\n  '{word}' not in voc!")
            results[word] = []

    # analogy experiments
    print(f"\n  analogy tests:")
    anls = [
        (["btech", "pg"], ["ug"], "UG:BTech :: PG:?"),
        (["teaching", "researcher"], ["professor"], "professor:teaching :: researcher:?"),
        (["exam", "thesis"], ["semester"], "semester:exam :: thesis:?"),  # buggy?
    ]

    a_res = []
    for positive, negative, description in anls:
        try:
            r_pos = [resolver(w) for w in positive]
            r_neg = [resolver(w) for w in negative]

            all_input_words = positive + negative
            all_resolved_words = r_pos + r_neg
            missing = [w for w, rw in zip(all_input_words, all_resolved_words) if rw is None]
            if missing:
                print(f"    {description}")
                print(f"      words not in vocab: {missing}")
                a_res.append({"description": description, "result": "words_missing", "missing": missing})
                continue

            if (r_pos != positive) or (r_neg != negative):  # seems to work
                print(f"    {description}")
                print(f"      resolved as positive={r_pos}, negative={r_neg}")
            else:
                print(f"    {description}")

            result = model.wv.most_similar(positive=r_pos, negative=r_neg, topn=3)
            for word, score in result:
                print(f"      {word:20s} {score:.4f}")
            a_res.append({"description": description, "result": [(w, float(s)) for w, s in result]})
        except Exception as e:
            print(f"    {description} → error: {e}")
            a_res.append({"description": description, "result": str(e)})

    return results, a_res

def sv_g(cb_m, sg_m, cb_r, sg_r):
    
    model_dir = os.path.join(O_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    # save models
    cb_m.save(os.path.join(model_dir, "gensim_cbow.model"))
    sg_m.save(os.path.join(model_dir, "gensim_skipgram.model"))

    # save embeddings as numpy
    words = list(cb_m.wv.key_to_index.keys())
    cbow_embeddings = np.array([cb_m.wv[w] for w in words])
    sg_embeddings = np.array([sg_m.wv[w] for w in words])

    np.save(os.path.join(model_dir, "gensim_cbow_embeddings.npy"), cbow_embeddings)
    np.save(os.path.join(model_dir, "gensim_sg_embeddings.npy"), sg_embeddings)

    # save word list
    with open(os.path.join(model_dir, "gensim_words.json"), 'w') as f:
        json.dump(words, f)

    print(f"\ngensim models and embeddings saved.")

def main():
    sentences = ld_corp()

    # train models with standard h_prm
    cb_m, sg_m = tr_gen(sentences, e_dim=100, window=3, neg=5, epochs=30)

    # analyze both models
    cb_r, cb_a = anl_m(cb_m, "Gensim CBOW")
    sg_r, sg_a = anl_m(sg_m, "Gensim Skip-gram")

    # save results
    sv_g(cb_m, sg_m, cb_r, sg_r)

    all_results = {
        "gensim_cbow": {
            "neighbors": {k: [(w, float(s)) for w, s in v] for k, v in cb_r.items()},
            "anls": cb_a
        },
        "gensim_skipgram": {
            "neighbors": {k: [(w, float(s)) for w, s in v] for k, v in sg_r.items()},
            "anls": sg_a
        }
    }
    with open(os.path.join(O_DIR, "gensim_analysis.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\ngensim comparison complete!")

if __name__ == "__main__":
    main()
