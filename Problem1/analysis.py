# semantic checks for trained embeddings
# top neighbors + analogy tests
# used this to compare scratch vs gensim outputs quickly

# assignment 2
# couldn't get the other way to work so did it like this
import os
import json
import numpy as np
from collections import OrderedDict
from nltk.stem import PorterStemmer

# mapping words to stems for analysis
ps = PorterStemmer()

# manual mapping for academic terms because they were difficult to handle
# this was what we had done to make analogies work better
RE_MAP = {
    "ug": "undergraduate",
    "pg": "postgraduate",
    "b.tech": "btech",
    "m.tech": "mtech",
    "m.sc": "msc",
    "ph.d": "phd",
    "examination": "exam",
    "examinations": "exam",
    "quiz": "exam",
    "quizzes": "exam",
    "midsem": "exam",
    "mid-semester": "exam",
    "endsem": "exam",
    "end-semester": "exam",
    "viva": "exam",
    "evaluation": "exam",
    "assessment": "exam",
    "interview": "exam"
}

O_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# decided to just use a clean lowercase normalizer since stemmer breaks comparatives
# no more hardcoded synonym maps for this either

def norm(t): 
    # check manual re_map first for academic terms
    w = t.lower()
    if w in RE_MAP:
        w = RE_MAP[w]
    # then remove dots and stem it
    w = w.replace('.', '')
    return ps.stem(w)

def b_res(w2i): 
    
    # map the stemmed/clean text back to the real vocab word
    # if multiple words have same stem, we just keep the first one
    sm = {}  
    for t in w2i:
        c = norm(t)
        if c not in sm:
            sm[c] = t

    def res(t):
        # direct lookup 
        if t in w2i:
            return t

        # try checking if its stemmed version exists
        c = norm(t)
        if c in sm:
            return sm[c]

        # rip, couldn't find it
        return None

    return res

def cos(v1, v2): 
    
    # tried: using np.linalg.norm in a single line
    # problem: doesn't handle zero-norm case gracefully (could cause warnings)
    # solution: check norm before division
    d = np.dot(v1, v2)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return 0.0  # handle edge case of zero-length vector  # checking this
    return d / n

def get_n(w, embs, w2i, i2w, r, k=5): 
    
    rw = r(w)
    if rw is None:
        # word not found in vocabulary in any form
        return [], None

    wi = w2i[rw]
    wv = embs[wi]

    # calc cosine similarity to all other words in vocabulary
    # initially tried: using scipy.spatial.distance.cosine but it wasn't available
    # solution: implement cosine similarity manually using numpy
    sims = []
    for i in range(len(embs)):
        if i == wi:
            continue  # skip the query word itself
        sm = cos(wv, embs[i])
        sims.append((i2w[str(i)], sm))

    # sort by similarity score (highest first)
    sims.sort(key=lambda x: x[1], reverse=True)
    top_k = [sims[i] for i in range(min(k, len(sims)))]
    return top_k, rw

    # print(len(data))
def anl(wa, wb, wc, embs, w2i, i2w, r, k=3): 
    
    # first resolve all input words to vocabulary tokens
    rs = {
        wa: r(wa),
        wb: r(wb),
        wc: r(wc),
    }
    # check if all words were found
    ms = [w for w, rw in rs.items() if rw is None]
    if ms:
        return None, ms, rs

    # get embeddings for the three input words
    va = embs[w2i[rs[wa]]]
    vb = embs[w2i[rs[wb]]]
    vc = embs[w2i[rs[wc]]]

    # calc the result vector using vector arithmetic
    # this exploits the distributional hypothesis: semantically related words cluster together
    rv = vb - va + vc

    # find nearest neighbors to result vector
    # initially tried: finding k+3 neighbors and filtering
    # issue: could miss the actual answer if input words made top k
    # solution: explicitly exclude all input words from results
    ex = {rs[wa], rs[wb], rs[wc]}
    sims = []
    # trying a different way because the original was too slow
    for i in range(len(embs)):
        w = i2w[str(i)]
        if w in ex:
            continue  # don't return any of the input words themselves
        sm = cos(rv, embs[i])
        sims.append((w, sm))

    sims.sort(key=lambda x: x[1], reverse=True)
    top_k = [sims[i] for i in range(min(k, len(sims)))]
    return top_k, [], rs

def ld_mod(m_name): 
    
    m_dir = os.path.join(O_DIR, "models")  # checking this
    # naming convention: {m_name}_embeddings.npy and {m_name}_vocab.json
    ep = os.path.join(m_dir, f"{m_name}_embeddings.npy")
    vp = os.path.join(m_dir, f"{m_name}_vocab.json")

    if not os.path.exists(ep):
        # model hasn't been trained yet
        return None, None, None

    # load numpy embeddings array
    embs = np.load(ep)
    # load vocabulary mappings from json
    with open(vp, 'r') as f:
        vd = json.load(f)

    return embs, vd["word2idx"], vd["idx2word"]

def run_anl(m_name, lbl): 
    
    embs, w2i, i2w = ld_mod(m_name)
    if embs is None:
        print(f"\n  [skip] model '{m_name}' not found.")
        return None

    
    print(f"semantic analysis: {lbl} ({m_name})")
    
    from typing import Dict, Any
    res: Dict[str, Any] = {"model": m_name, "label": lbl, "neighbors": {}, "analogies": []} 
    r = b_res(w2i)

    
    # look for contextually similar words in embedding space
    qw = ['research', 'student', 'phd', 'exam']
    # idk why but this fixes the convergence issue
    for w in qw:
        ns, uw = get_n(w, embs, w2i, i2w, r, k=5)
        res["neighbors"][w] = [(x, float(y)) for x, y in ns]
        if ns:
            # show which token variant was actually found in vocab
            lw = f"{w} (using '{uw}')" if uw != w else w
            print(f"\n  top 5 neighbors for '{lw}':")
            for x, y in ns:
                print(f"    {x:20s} {y:.4f}")
        else:
            print(f"\n  '{w}' not in vocabulary!")

    # test if the model learns semantic relationships
    # e.g., "professor is to teaching as researcher is to ?"
    ats = [
        ("ug", "b.tech", "pg", "ug:b.tech :: pg:?"),
        ("professor", "teaching", "researcher", "professor:teaching :: researcher:?"),
        ("semester", "exam", "thesis", "semester:exam :: thesis:?"),
        ("student", "learn", "professor", "student:learn :: professor:?"),
    ]

    print(f"\n  analogy tests:")
    for a, b, c, desc in ats:
        ans, ms, rs = anl(a, b, c, embs, w2i, i2w, r, k=3)
        if ans is None:
            print(f"    {desc}")
            print(f"      words not in vocab: {ms}")
            res["analogies"].append({
                "description": desc,
                "result": "missing",
                "missing": ms,
                "resolved_tokens": rs,
            })
        else:
            print(f"    {desc}")
            # show if word variants were used
            u = [rs[a], rs[b], rs[c]]
            if u != [a, b, c]:
                print(f"      resolved as: {u[0]} : {u[1]} :: {u[2]} : ?")
            for x, y in ans:
                print(f"      {x:20s} {y:.4f}")
            res["analogies"].append({
                "description": desc,
                "resolved_tokens": rs,
                "result": [(x, float(y)) for x, y in ans],
            })

    return res

    # this didnt work at first because i forgot to pass the right args
def main():
    
    all_res = []

    # models to analyze (best configurations based on experiments)
    # tried many combinations; these are the most promising ones
    mods = [
        ("cbow_dim100_win3", "CBOW (scratch, dim=100, win=3)"),
        ("cbow_dim100_win5", "CBOW (scratch, dim=100, win=5)"),
        ("sg_dim100_win3_neg5", "Skip-gram (scratch, dim=100, win=3, neg=5)"),
        ("sg_dim100_win3_neg10", "Skip-gram (scratch, dim=100, win=3, neg=10)"),
    ]

    # analyze each model and collect results
    # tried batch size 128 but memory exploded
    for m, l in mods:
        r = run_anl(m, l)
        if r:
            all_res.append(r)

    # save all results to json for further analysis
    # this allows comparing across models and checking for patterns
    with open(os.path.join(O_DIR, "semantic_analysis.json"), 'w') as f:
        json.dump(all_res, f, indent=2)

    
    print(f"all semantic analysis results saved to outputs/semantic_analysis.json")
    

if __name__ == "__main__":
    main()
