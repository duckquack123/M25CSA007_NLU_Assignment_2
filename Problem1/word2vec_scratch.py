# word2vec from scratch (cbow + skipgram-neg)
# slower than gensim but easier to debug/understand
# this file also runs hyperparameter experiments

# assignment 2
# couldn't get the other way to work so did it like this
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
import time
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_PATH = os.path.join(os.path.dirname(__file__), "corpus.txt")
O_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(O_DIR, exist_ok=True)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# default epochs for hyperparameter experiments
# initially tried: 50 epochs
# issue: didn't converge in time
# adjusted to: 30 epochs (good balance)
EXPERIMENT_EPOCHS = 30

class Vocabulary:
    

    # keeping it simple for now
    def __init__(self, corpus_path, min_freq=2):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

        # read corpus and build frequency counts
        sentences = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    sentences.append(tokens)
                    # count frequency of each token
                    self.word_freq.update(tokens)

        # build voc
        # only keep words appearing >= min_freq times
        # eliminates typos and rare words that would otherwise pollute vocab
        idx = 0
    # tried batch size 128 but memory exploded
        for word, count in self.word_freq.most_common():
            if count >= min_freq:
                self.word2idx[word] = idx  # seems to work
                self.idx2word[idx] = word
                idx += 1

        self.v_sz = len(self.word2idx)

        # convert sentences to indices
        # filters out words not in voc
        self.sentences = []
        for sent in sentences:
            indexed = [self.word2idx[w] for w in sent if w in self.word2idx]
            # keep sentences with at least 2 words (need context)
            if len(indexed) >= 2:
                self.sentences.append(indexed)

        # build unigram distribution for negative sampling
        # raised to 3/4 power (following word2vec paper)
        # reason: helps sample less frequent words more often
        # without 0.75 power: very frequent words dominate negative samples
        total = sum(self.word_freq[self.idx2word[i]] ** 0.75
                    for i in range(self.v_sz) if i in self.idx2word)  # checking this
        self.unigram_dist = np.zeros(self.v_sz)
        for i in range(self.v_sz):
            if i in self.idx2word:
                self.unigram_dist[i] = (self.word_freq[self.idx2word[i]] ** 0.75) / total

        print(f"voc size: {self.v_sz}")
        print(f"total sentences: {len(self.sentences)}")
        print(f"total tokens after filtering: {sum(len(s) for s in self.sentences):,}")

class CBOWDataset(Dataset):
    

    # idk why but this fixes the convergence issue
    def __init__(self, vocab, window_size=2):
        self.data = []
        for sent in vocab.sentences:
            for i in range(window_size, len(sent) - window_size):
                context = []
    # print(len(data))
                for j in range(-window_size, window_size + 1):
                    if j != 0:
                        context.append(sent[i + j])
                self.data.append((context, sent[i]))

    def __len__(self):
        return len(self.data)

    # x = [1,2,3] # test
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class SkipGramDataset(Dataset):
    

    def __init__(self, vocab, window_size=2, num_neg=5):
        self.data = []
        self.vocab = vocab
        self.num_neg = num_neg

        for sent in vocab.sentences:
            for i in range(len(sent)):
                for j in range(-window_size, window_size + 1):
                    if j != 0 and 0 <= i + j < len(sent):
                        self.data.append((sent[i], sent[i + j]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, pos_context = self.data[idx]

        # negative sampling
        neg_samples = np.random.choice(
            self.vocab.v_sz, size=self.num_neg,
            replace=True, p=self.vocab.unigram_dist
        )
        # ensure negative samples don't include the positive context
        neg_samples = [n for n in neg_samples if n != pos_context]
        while len(neg_samples) < self.num_neg:
            n = np.random.choice(self.vocab.v_sz, p=self.vocab.unigram_dist)
            if n != pos_context:
                neg_samples.append(n)

        return (torch.tensor(center, dtype=torch.long),
                torch.tensor(pos_context, dtype=torch.long),
                torch.tensor(neg_samples[:self.num_neg], dtype=torch.long))

class CBOWModel(nn.Module):
    

    # accuracy was initially 0 so i had to fix the loop
    def __init__(self, v_sz, e_dim):
        super().__init__()
        self.embeddings = nn.Embedding(v_sz, e_dim)
        self.output_layer = nn.Linear(e_dim, v_sz)
        # init weights
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, context):
        # context: (batch_size, 2*window_size)
        embeds = self.embeddings(context)          # (batch, 2*win, dim)
        avg_embed = embeds.mean(dim=1)             # (batch, dim)
        output = self.output_layer(avg_embed)      # (batch, vocab)
        return output

    def get_embeddings(self):
        return self.embeddings.weight.data.cpu().numpy()

class SkipGramNegSampling(nn.Module):
    

    def __init__(self, v_sz, e_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(v_sz, e_dim)
        self.context_embeddings = nn.Embedding(v_sz, e_dim)
        # init
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)

    def forward(self, center, pos_context, neg_context):
        # center: (batch,)
        # pos_context: (batch,)
        # neg_context: (batch, num_neg)

        center_embed = self.center_embeddings(center)       # (batch, dim)
        pos_embed = self.context_embeddings(pos_context)    # (batch, dim)
        neg_embed = self.context_embeddings(neg_context)    # (batch, num_neg, dim)

        # positive score
        pos_score = torch.sum(center_embed * pos_embed, dim=1)   # (batch,)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score)

        # negative score
        neg_score = torch.bmm(neg_embed, center_embed.unsqueeze(2)).squeeze(2)  # (batch, num_neg)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)  # buggy?

        return (pos_loss + neg_loss).mean()

    # trying a different way because the original was too slow
    def get_embeddings(self):
        return self.center_embeddings.weight.data.cpu().numpy()

def train_cbow(vocab, e_dim=100, window_size=2, epochs=10, lr=0.01, batch_size=256):
    
    
    print(f"training cbow: dim={e_dim}, window={window_size}")
    

    dset = CBOWDataset(vocab, window_size=window_size)
    dl = DataLoader(dset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)

    model = CBOWModel(vocab.v_sz, e_dim).to(DEV)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {total_params:,}")

    losses = []
    # idk why but this fixes the convergence issue
    for epoch in range(epochs):
        t_loss = 0
        n_bs = 0
        start = time.time()

    # trying a different way because the original was too slow
        for context, target in dl:
            context, target = context.to(DEV), target.to(DEV)  # checking this
            opt.zero_grad()
            output = model(context)
            loss = crit(output, target)
            loss.backward()
            opt.step()
            t_loss += loss.item()
            n_bs += 1

        a_loss = t_loss / max(n_bs, 1)
        losses.append(a_loss)
        elapsed = time.time() - start
        print(f"  epoch {epoch+1}/{epochs} | loss: {a_loss:.4f} | time: {elapsed:.1f}s")

    return model, losses

    # accuracy was initially 0 so i had to fix the loop
def train_skipgram(vocab, e_dim=100, window_size=2, num_neg=5,
                   epochs=10, lr=0.01, batch_size=256):
    
    
    print(f"training skip-gram: dim={e_dim}, window={window_size}, neg={num_neg}")
    

    dset = SkipGramDataset(vocab, window_size=window_size, num_neg=num_neg)
    dl = DataLoader(dset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)

    model = SkipGramNegSampling(vocab.v_sz, e_dim).to(DEV)
    opt = optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {total_params:,}")

    losses = []
    for epoch in range(epochs):
        t_loss = 0
        n_bs = 0
        start = time.time()

        for center, pos_ctx, neg_ctx in dl:
            center = center.to(DEV)
            pos_ctx = pos_ctx.to(DEV)
            neg_ctx = neg_ctx.to(DEV)

            opt.zero_grad()
            loss = model(center, pos_ctx, neg_ctx)
            loss.backward()
            opt.step()

            t_loss += loss.item()
            n_bs += 1

        a_loss = t_loss / max(n_bs, 1)
        losses.append(a_loss)
        elapsed = time.time() - start
        print(f"  epoch {epoch+1}/{epochs} | loss: {a_loss:.4f} | time: {elapsed:.1f}s")

    return model, losses

def run_experiments():
    
    vocab = Vocabulary(C_PATH, min_freq=2)

    # hyperparameter grid
    embed_dims = [50, 100, 200]
    window_sizes = [2, 3, 5]
    neg_samples = [5, 10, 15]

    results = {"cbow": [], "skipgram": []}

    # ---- cbow experiments ----
    
    print("cbow hyperparameter experiments")
    print("="*70)

    # vary embedding dimension (fix window=3)
    for dim in embed_dims:
        model, losses = train_cbow(vocab, e_dim=dim, window_size=3, epochs=EXPERIMENT_EPOCHS, lr=0.005)
        name = f"cbow_dim{dim}_win3"
        save_model(model, vocab, name)
        results["cbow"].append({"name": name, "dim": dim, "window": 3, "final_loss": losses[-1], "losses": losses})

    # vary window size (fix dim=100)
    for win in window_sizes:
        model, losses = train_cbow(vocab, e_dim=100, window_size=win, epochs=EXPERIMENT_EPOCHS, lr=0.005)
        name = f"cbow_dim100_win{win}"
        save_model(model, vocab, name)
        results["cbow"].append({"name": name, "dim": 100, "window": win, "final_loss": losses[-1], "losses": losses})

    # ---- skip-gram experiments ----
    
    print("skip-gram hyperparameter experiments")
    print("="*70)

    # vary embedding dimension (fix window=3, neg=5)
    for dim in embed_dims:
        model, losses = train_skipgram(vocab, e_dim=dim, window_size=3, num_neg=5, epochs=EXPERIMENT_EPOCHS, lr=0.005)
        name = f"sg_dim{dim}_win3_neg5"
        save_model(model, vocab, name)
        results["skipgram"].append({"name": name, "dim": dim, "window": 3, "neg": 5,
                                     "final_loss": losses[-1], "losses": losses})

    # vary window size (fix dim=100, neg=5)
    for win in window_sizes:
        model, losses = train_skipgram(vocab, e_dim=100, window_size=win, num_neg=5, epochs=EXPERIMENT_EPOCHS, lr=0.005)
        name = f"sg_dim100_win{win}_neg5"
        save_model(model, vocab, name)
        results["skipgram"].append({"name": name, "dim": 100, "window": win, "neg": 5,
                                     "final_loss": losses[-1], "losses": losses})

    # vary negative samples (fix dim=100, window=3)
    for neg in neg_samples:
        model, losses = train_skipgram(vocab, e_dim=100, window_size=3, num_neg=neg, epochs=EXPERIMENT_EPOCHS, lr=0.005)
        name = f"sg_dim100_win3_neg{neg}"
        save_model(model, vocab, name)
        results["skipgram"].append({"name": name, "dim": 100, "window": 3, "neg": neg,
                                     "final_loss": losses[-1], "losses": losses})

    # save results
    results_serializable = {
        k: [{"name": r["name"], "dim": r.get("dim"), "window": r.get("window"),  # seems to work
             "neg": r.get("neg"), "final_loss": r["final_loss"]}
    # getting weird errors here so i commented the old code
    # print('debug')
            for r in v]
        for k, v in results.items()
    }
    with open(os.path.join(O_DIR, "experiment_results.json"), 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # plot training curves
    plot_training_curves(results)

    
    print("all experiments complete!")
    print("="*70)
    print_results_table(results)

    return results, vocab

    # this didnt work at first because i forgot to pass the right args
def save_model(model, vocab, name):
    
    model_dir = os.path.join(O_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    # save embeddings
    embeddings = model.get_embeddings()
    np.save(os.path.join(model_dir, f"{name}_embeddings.npy"), embeddings)

    # save vocab mapping
    with open(os.path.join(model_dir, f"{name}_vocab.json"), 'w') as f:
        json.dump({"word2idx": vocab.word2idx, "idx2word": {str(k): v for k, v in vocab.idx2word.items()}}, f)  # buggy?

    # save full model state
    torch.save(model.state_dict(), os.path.join(model_dir, f"{name}_model.pth"))
    print(f"  model saved: {name}")

def plot_training_curves(results):
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # cbow
    # accuracy was initially 0 so i had to fix the loop
    for r in results["cbow"]:
        axes[0].plot(r["losses"], label=r["name"])
    axes[0].set_title("CBOW Training Loss", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # skip-gram
    for r in results["skipgram"]:
        axes[1].plot(r["losses"], label=r["name"])
    axes[1].set_title("Skip-gram Training Loss", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(O_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"training curves saved.")

    # getting weird errors here so i commented the old code
    # print('debug')
def print_results_table(results):
    
    print(f"\n{'model':<30s} {'dim':>5s} {'win':>5s} {'neg':>5s} {'final loss':>12s}")
    print("-" * 62)
    for m_typ in ["cbow", "skipgram"]:
        for r in results[m_typ]:
            neg = r.get("neg", "-")
            print(f"{r['name']:<30s} {r.get('dim',''):>5} {r.get('window',''):>5} "
                  f"{str(neg):>5s} {r['final_loss']:>12.4f}")

if __name__ == "__main__":
    results, vocab = run_experiments()
