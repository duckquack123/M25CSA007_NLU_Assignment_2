# embedding plots (pca + tsne)
# mostly for sanity check, not strict metric

# assignment 2
# couldn't get the other way to work so did it like this
import os
import json
import inspect
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use agg backend for non-gui rendering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

O_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(O_DIR, exist_ok=True)

    # took forever to run on cpu
def ld_m(m_nm):
    # loads saved embeddings + vocab
    model_dir = os.path.join(O_DIR, "models")
    emb_path = os.path.join(model_dir, f"{m_nm}_embeddings.npy")
    vocab_path = os.path.join(model_dir, f"{m_nm}_vocab.json")

    if not os.path.exists(emb_path):
        print(f"[skip] model not found: {m_nm}")
        return None, None, None

    # load embeddings as numpy array
    embeddings = np.load(emb_path)
    # load voc mappings
    with open(vocab_path, 'r') as f:
        v_dat = json.load(f)
    return embeddings, v_dat["word2idx"], v_dat["idx2word"]

def get_semantic_groups():
    
    return {
        "Academic Programs": ["btech", "mtech", "phd", "msc", "undergraduate",
                               "postgraduate", "degree", "diploma", "programme", "course"],
        "Research": ["research", "paper", "publication", "journal", "conference",
                     "thesis", "dissertation", "project", "innovation", "lab"],
        "People": ["student", "professor", "faculty", "teacher", "researcher",
                   "scholar", "dean", "director", "hod", "staff"],
        "Departments": ["cse", "electrical", "mechanical", "physics", "chemistry",
                        "mathematics", "computer", "science", "engineering", "department"],
        "Assessment": ["exam", "grade", "marks", "semester", "credit",
                       "evaluation", "assessment", "test", "cgpa", "sgpa"],
    }

def plot_embeddings_2d(embeddings, word2idx, idx2word, method='pca', model_label='',
                        save_name='', semantic_groups=None):  # seems to work
    
    if semantic_groups is None:
        semantic_groups = get_semantic_groups()

    # collect words from semantic groups
    selected_words = []
    word_groups = []
    word_colors = []

    colors = plt.cm.Set2(np.linspace(0, 1, len(semantic_groups)))
    color_map = {}

    # map words to groups and colors
    for i, (group_name, words) in enumerate(semantic_groups.items()):
        for word in words:
            if word in word2idx:  # only if word exists in learned vocab
                selected_words.append(word)
                word_groups.append(group_name)
                color_map[group_name] = colors[i]
                word_colors.append(colors[i])

    # need minimum words for meaningful visualization
    if len(selected_words) < 5:
        print(f"  [skip] not enough words in voc for {model_label}")
        return

    # get embeddings for selected words
    selected_embeddings = np.array([embeddings[word2idx[w]] for w in selected_words])

    # reduce dimensions
    if method == 'pca':
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(selected_embeddings)
        explained_var = reducer.explained_variance_ratio_
        title_suffix = f"(PCA, var={sum(explained_var)*100:.1f}%)"
    else:
        perplexity = min(30, len(selected_words) - 1)
        tsne_kwargs = {  # seems to work
            'n_components': 2,
            'perplexity': perplexity,
            'random_state': 42,  # checking this
            'learning_rate': 'auto',
            'init': 'pca',
        }
        # handle sklearn api changes: older versions use n_iter, newer versions use max_iter
        if 'max_iter' in inspect.signature(TSNE.__init__).parameters:
            tsne_kwargs['max_iter'] = 1000
        else:
            tsne_kwargs['n_iter'] = 1000

        reducer = TSNE(**tsne_kwargs)  # checking this
        coords = reducer.fit_transform(selected_embeddings)
        title_suffix = "(t-SNE)"

    # plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # plot points by group
    # took forever to run on cpu
    for group_name in semantic_groups:
        mask = [g == group_name for g in word_groups]
        if not any(mask):
            continue
        group_coords = coords[mask]
        ax.scatter(group_coords[:, 0], group_coords[:, 1],
                   c=[color_map[group_name]], s=100, alpha=0.8,
                   label=group_name, edgecolors='white', linewidth=0.5)

    # add word labels
    for i, word in enumerate(selected_words):
        ax.annotate(word, (coords[i, 0], coords[i, 1]),
                    fontsize=9, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points',
                    alpha=0.85)  # checking this

    ax.set_title(f"{model_label} — Word Embeddings {title_suffix}",
                 fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)

    plt.tight_layout()
    filepath = os.path.join(O_DIR, f"{save_name}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {filepath}")

def plot_comparison(cbow_name, sg_name):
    
    cbow_emb, cbow_w2i, cbow_i2w = ld_m(cbow_name)
    sg_emb, sg_w2i, sg_i2w = ld_m(sg_name)

    if cbow_emb is None or sg_emb is None:
        print("[skip] models not found for comparison.")
        return

    semantic_groups = get_semantic_groups()

    # pca plots
    print("\ngenerating pca visualizations...")
    plot_embeddings_2d(cbow_emb, cbow_w2i, cbow_i2w, method='pca',
                       model_label='CBOW (Scratch)',
                       save_name=f'pca_{cbow_name}',
                       semantic_groups=semantic_groups)
    plot_embeddings_2d(sg_emb, sg_w2i, sg_i2w, method='pca',
                       model_label='Skip-gram (Scratch)',
                       save_name=f'pca_{sg_name}',
                       semantic_groups=semantic_groups)

    # t-sne plots
    print("\ngenerating t-sne visualizations...")
    plot_embeddings_2d(cbow_emb, cbow_w2i, cbow_i2w, method='tsne',
                       model_label='CBOW (Scratch)',
                       save_name=f'tsne_{cbow_name}',
                       semantic_groups=semantic_groups)
    plot_embeddings_2d(sg_emb, sg_w2i, sg_i2w, method='tsne',
                       model_label='Skip-gram (Scratch)',
                       save_name=f'tsne_{sg_name}',
                       semantic_groups=semantic_groups)

def main():
    
    print("="*60)
    print("word embedding visualization")
    print("="*60)

    # compare best cbow and skip-gram models
    plot_comparison("cbow_dim100_win3", "sg_dim100_win3_neg5")

    # also visualize other hyperparameter configs if available
    for name in ["cbow_dim50_win3", "cbow_dim200_win3",
                 "sg_dim50_win3_neg5", "sg_dim200_win3_neg5"]:
        emb, w2i, i2w = ld_m(name)
        if emb is not None:
            print(f"\nvisualization for {name}:")
            plot_embeddings_2d(emb, w2i, i2w, method='pca',
                               model_label=name, save_name=f'pca_{name}')

    
    print("all visualizations saved to outputs/")
    

if __name__ == "__main__":
    main()
