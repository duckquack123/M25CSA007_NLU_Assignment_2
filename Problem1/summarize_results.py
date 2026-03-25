import os
import json
import glob

# Finalize Word2Vec Scratch Results
# Run this if you stopped training early to get the experiment_results.json deliverable.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
O_DIR = os.path.join(BASE_DIR, "outputs")
M_DIR = os.path.join(O_DIR, "models")

def get_params(name):
    parts = name.split("_")
    params = {"dim": 100, "win": 3, "neg": 5}
    for p in parts:
        if p.startswith("dim"): params["dim"] = int(p.replace("dim", ""))
        elif p.startswith("win"): params["win"] = int(p.replace("win", ""))
        elif p.startswith("neg"): params["neg"] = int(p.replace("neg", ""))
    return params["dim"], params["win"], params["neg"]

def main():
    print(f"Assignement 2 - Problem 1: Result Recovery")
    print(f"Scanning models in: {M_DIR}")
    
    files = glob.glob(os.path.join(M_DIR, "*_embeddings.npy"))
    files = [f for f in files if "gensim" not in f]
    
    if not files:
        print("No trained models found!")
        return

    results = {"cbow": [], "skipgram": []}
    for f in files:
        name = os.path.basename(f).replace("_embeddings.npy", "")
        m_type = "cbow" if "cbow" in name else "skipgram"
        dim, win, neg = get_params(name)
        results[m_type].append({"name": name, "dim": dim, "window": win, "neg": neg, "final_loss": "(See Logs)"})

    results["cbow"].sort(key=lambda x: (x["dim"], x["window"]))
    results["skipgram"].sort(key=lambda x: (x["dim"], x["window"], x["neg"]))

    res_path = os.path.join(O_DIR, "experiment_results.json")
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print(f"{'Recovered Model':<30s} | {'Dim':^5s} | {'Win':^5s} | {'Neg':^5s}")
    print("-" * 70)
    for m in results["cbow"] + results["skipgram"]:
        print(f"{m['name']:<30s} | {str(m['dim']):^5s} | {str(m['window']):^5s} | {str(m['neg']):^5s}")
    print("=" * 70)
    print(f"\n[OK] 'experiment_results.json' generated. You can now finish your report.")

if __name__ == "__main__":
    main()
