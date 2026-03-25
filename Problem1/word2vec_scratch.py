# word2vec from scratch (cbow + skipgram-neg)
# pure numpy implementation (no pytorch)
# kept explicit on purpose so each math step is inspectable while debugging

import os
import json
import time
import pickle
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_PATH = os.path.join(os.path.dirname(__file__), "corpus.txt")
O_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(O_DIR, exist_ok=True)

EXPERIMENT_EPOCHS = 30


class Vocabulary:
    def __init__(self, corpus_path, min_freq=2):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()

        # read once and keep raw tokenized lines; later we map to ids
        sentences = []
        # Failure point: this will raise FileNotFoundError if corpus.txt is missing.
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    sentences.append(tokens)
                    self.word_freq.update(tokens)

        # sort by frequency and then cut with min_freq
        idx = 0
        for word, count in self.word_freq.most_common():
            if count >= min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        self.v_sz = len(self.word2idx)

        # convert sentence tokens -> integer ids for fast training loops
        self.sentences = []
        for sent in sentences:
            indexed = [self.word2idx[w] for w in sent if w in self.word2idx]
            if len(indexed) >= 2:
                self.sentences.append(indexed)

        # Word2Vec negative sampling distribution (unigram^0.75).
        # Failure point: if vocabulary becomes empty after filtering, total can be zero.
        total = sum(
            self.word_freq[self.idx2word[i]] ** 0.75
            for i in range(self.v_sz)
            if i in self.idx2word
        )
        self.unigram_dist = np.zeros(self.v_sz, dtype=np.float64)
        for i in range(self.v_sz):
            if i in self.idx2word:
                self.unigram_dist[i] = (self.word_freq[self.idx2word[i]] ** 0.75) / total

        print(f"voc size: {self.v_sz}")
        print(f"total sentences: {len(self.sentences)}")
        print(f"total tokens after filtering: {sum(len(s) for s in self.sentences):,}")


class CBOWDataset:
    def __init__(self, vocab, window_size=2):
        # Build (context -> target) training pairs for CBOW.
        # yes this can get big in memory, but iteration becomes simple + fast
        self.contexts = []
        self.targets = []
        for sent in vocab.sentences:
            for i in range(window_size, len(sent) - window_size):
                ctx = []
                for j in range(-window_size, window_size + 1):
                    if j != 0:
                        ctx.append(sent[i + j])
                self.contexts.append(ctx)
                self.targets.append(sent[i])

        self.contexts = np.asarray(self.contexts, dtype=np.int64)
        self.targets = np.asarray(self.targets, dtype=np.int64)

    def __len__(self):
        return len(self.targets)


class SkipGramDataset:
    def __init__(self, vocab, window_size=2):
        # Build (center -> positive context) training pairs for Skip-gram.
        # skip-gram expands more aggressively than cbow, expected behavior
        self.centers = []
        self.positives = []
        for sent in vocab.sentences:
            for i in range(len(sent)):
                for j in range(-window_size, window_size + 1):
                    if j != 0 and 0 <= i + j < len(sent):
                        self.centers.append(sent[i])
                        self.positives.append(sent[i + j])

        self.centers = np.asarray(self.centers, dtype=np.int64)
        self.positives = np.asarray(self.positives, dtype=np.int64)

    def __len__(self):
        return len(self.centers)


def xavier_uniform(shape):
    # Xavier init keeps activations/gradients in a stable range.
    # tried random normal earlier; this is usually more stable for training
    fan_in, fan_out = shape[0], shape[1]
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-bound, bound, size=shape).astype(np.float32)


def sigmoid(x):
    # Clip input to avoid overflow in exp for large magnitude logits.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyAdam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        # Failure point: parameter and gradient shapes must match exactly.
        # if loss becomes nan, first thing to check is lr (too high is common)
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class CBOWModelScratch:
    def __init__(self, v_sz, e_dim):
        self.v_sz = v_sz
        self.e_dim = e_dim
        self.embeddings = xavier_uniform((v_sz, e_dim))
        self.output_weight = xavier_uniform((e_dim, v_sz))
        self.output_bias = np.zeros(v_sz, dtype=np.float32)

    def parameters(self):
        return [self.embeddings, self.output_weight, self.output_bias]

    def forward(self, context_batch):
        # Gather context vectors, average them, then project to vocab logits.
        # classic cbow: mean(context words) predicts center word
        embeds = self.embeddings[context_batch]
        avg_embed = embeds.mean(axis=1)
        logits = avg_embed @ self.output_weight + self.output_bias
        return avg_embed, logits

    def backward(self, context_batch, avg_embed, logits, targets):
        batch_size = context_batch.shape[0]

        # Numerically stable softmax.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        # Add epsilon to avoid log(0) when probabilities underflow.
        # tiny epsilon saves us from random numerical crashes mid-run
        loss = -np.log(probs[np.arange(batch_size), targets] + 1e-12).mean()

        dlogits = probs
        dlogits[np.arange(batch_size), targets] -= 1.0
        dlogits /= batch_size

        grad_out_w = avg_embed.T @ dlogits
        grad_out_b = dlogits.sum(axis=0)
        davg = dlogits @ self.output_weight.T

        ctx_len = context_batch.shape[1]
        # Each context token receives an equal share of gradient from the mean op.
        dembed = np.repeat(davg / ctx_len, ctx_len, axis=0)

        # np.add.at handles repeated word indices safely during gradient accumulation.
        grad_emb = np.zeros_like(self.embeddings)
        np.add.at(grad_emb, context_batch.reshape(-1), dembed)

        return float(loss), [grad_emb, grad_out_w.astype(np.float32), grad_out_b.astype(np.float32)]

    def get_embeddings(self):
        return self.embeddings

    def state_dict(self):
        return {
            "embeddings": self.embeddings,
            "output_weight": self.output_weight,
            "output_bias": self.output_bias,
        }


class SkipGramNegSamplingScratch:
    def __init__(self, v_sz, e_dim):
        self.v_sz = v_sz
        self.e_dim = e_dim
        self.center_embeddings = xavier_uniform((v_sz, e_dim))
        self.context_embeddings = xavier_uniform((v_sz, e_dim))

    def parameters(self):
        return [self.center_embeddings, self.context_embeddings]

    def forward(self, center_batch, pos_batch, neg_batch):
        # Score positive and negative pairs with dot products.
        # positives should score high, sampled negatives should score low
        center = self.center_embeddings[center_batch]
        pos = self.context_embeddings[pos_batch]
        neg = self.context_embeddings[neg_batch]

        pos_score = np.sum(center * pos, axis=1)
        neg_score = np.einsum('bnd,bd->bn', neg, center)

        return center, pos, neg, pos_score, neg_score

    def backward(self, center_batch, pos_batch, neg_batch, cache):
        center, pos, neg, pos_score, neg_score = cache
        batch_size = center.shape[0]

        pos_sig = sigmoid(pos_score)
        neg_sig = sigmoid(neg_score)

        # Epsilon avoids NaNs from log(0) in early unstable training.
        # skip-gram is noisier than cbow, so this guard matters in first epochs
        loss = -np.mean(np.log(pos_sig + 1e-12) + np.sum(np.log(sigmoid(-neg_score) + 1e-12), axis=1))

        dpos = (pos_sig - 1.0) / batch_size
        dneg = neg_sig / batch_size

        grad_center = dpos[:, None] * pos + np.sum(dneg[:, :, None] * neg, axis=1)
        grad_pos = dpos[:, None] * center
        grad_neg = dneg[:, :, None] * center[:, None, :]

        grad_center_emb = np.zeros_like(self.center_embeddings)
        grad_context_emb = np.zeros_like(self.context_embeddings)

        np.add.at(grad_center_emb, center_batch, grad_center)
        np.add.at(grad_context_emb, pos_batch, grad_pos)
        np.add.at(grad_context_emb, neg_batch.reshape(-1), grad_neg.reshape(-1, self.e_dim))

        return float(loss), [grad_center_emb.astype(np.float32), grad_context_emb.astype(np.float32)]

    def get_embeddings(self):
        return self.center_embeddings

    def state_dict(self):
        return {
            "center_embeddings": self.center_embeddings,
            "context_embeddings": self.context_embeddings,
        }


def iter_minibatches(size, batch_size, shuffle=True):
    # Yields index slices to avoid copying full dataset tensors.
    # intentionally tiny helper to keep training loops readable
    idx = np.arange(size)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, size, batch_size):
        end = min(start + batch_size, size)
        yield idx[start:end]


def sample_negative(unigram_dist, num_neg, batch_size, positive_batch):
    v_sz = unigram_dist.shape[0]
    neg = np.random.choice(v_sz, size=(batch_size, num_neg), replace=True, p=unigram_dist)
    # Keep re-sampling until negatives differ from positive target.
    # Failure point: if v_sz == 1, this loop cannot resolve conflicts.
    # practically fine for this corpus, but worth documenting for toy inputs
    conflict = (neg == positive_batch[:, None])
    while np.any(conflict):
        neg[conflict] = np.random.choice(v_sz, size=conflict.sum(), replace=True, p=unigram_dist)
        conflict = (neg == positive_batch[:, None])
    return neg.astype(np.int64)


def train_cbow(vocab, e_dim=100, window_size=2, epochs=10, lr=0.01, batch_size=256):
    print(f"training cbow: dim={e_dim}, window={window_size}")

    dset = CBOWDataset(vocab, window_size=window_size)
    model = CBOWModelScratch(vocab.v_sz, e_dim)
    opt = NumpyAdam(model.parameters(), lr=lr)

    total_params = sum(p.size for p in model.parameters())
    print(f"trainable parameters: {total_params:,}")

    losses = []
    # straightforward loop: batch -> forward -> backward -> adam step
    for epoch in range(epochs):
        t_loss = 0.0
        n_bs = 0
        start = time.time()

        for batch_idx in iter_minibatches(len(dset), batch_size, shuffle=True):
            context = dset.contexts[batch_idx]
            target = dset.targets[batch_idx]

            avg_embed, logits = model.forward(context)
            loss, grads = model.backward(context, avg_embed, logits, target)
            opt.step(grads)

            t_loss += loss
            n_bs += 1

        a_loss = t_loss / max(n_bs, 1)
        losses.append(a_loss)
        elapsed = time.time() - start
        print(f"  epoch {epoch+1}/{epochs} | loss: {a_loss:.4f} | time: {elapsed:.1f}s")

    return model, losses


def train_skipgram(vocab, e_dim=100, window_size=2, num_neg=5, epochs=10, lr=0.01, batch_size=256):
    print(f"training skip-gram: dim={e_dim}, window={window_size}, neg={num_neg}")

    dset = SkipGramDataset(vocab, window_size=window_size)
    model = SkipGramNegSamplingScratch(vocab.v_sz, e_dim)
    opt = NumpyAdam(model.parameters(), lr=lr)

    total_params = sum(p.size for p in model.parameters())
    print(f"trainable parameters: {total_params:,}")

    losses = []
    # same skeleton as cbow, but with sampled negatives each batch
    for epoch in range(epochs):
        t_loss = 0.0
        n_bs = 0
        start = time.time()

        for batch_idx in iter_minibatches(len(dset), batch_size, shuffle=True):
            center = dset.centers[batch_idx]
            pos_ctx = dset.positives[batch_idx]
            neg_ctx = sample_negative(vocab.unigram_dist, num_neg, len(batch_idx), pos_ctx)

            cache = model.forward(center, pos_ctx, neg_ctx)
            loss, grads = model.backward(center, pos_ctx, neg_ctx, cache)
            opt.step(grads)

            t_loss += loss
            n_bs += 1

        a_loss = t_loss / max(n_bs, 1)
        losses.append(a_loss)
        elapsed = time.time() - start
        print(f"  epoch {epoch+1}/{epochs} | loss: {a_loss:.4f} | time: {elapsed:.1f}s")

    return model, losses


def run_experiments():
    vocab = Vocabulary(C_PATH, min_freq=2)

    embed_dims = [50, 100, 200]
    window_sizes = [2, 3, 5]
    neg_samples = [5, 10, 15]

    results = {"cbow": [], "skipgram": []}

    print("cbow hyperparameter experiments")
    print("=" * 70)
    # This full grid is compute-heavy in NumPy and can take a long time.
    # stage 1: vary embedding dimension (fixed window)
    for dim in embed_dims:
        model, losses = train_cbow(vocab, e_dim=dim, window_size=3, epochs=EXPERIMENT_EPOCHS, lr=0.005)
        name = f"cbow_dim{dim}_win3"
        save_model(model, vocab, name)
        results["cbow"].append({"name": name, "dim": dim, "window": 3, "final_loss": losses[-1], "losses": losses})

    # stage 2: vary context window (fixed dim)
    for win in window_sizes:
        model, losses = train_cbow(vocab, e_dim=100, window_size=win, epochs=EXPERIMENT_EPOCHS, lr=0.005)
        name = f"cbow_dim100_win{win}"
        save_model(model, vocab, name)
        results["cbow"].append({"name": name, "dim": 100, "window": win, "final_loss": losses[-1], "losses": losses})

    print("skip-gram hyperparameter experiments")
    print("=" * 70)
    # stage 3: vary embedding dimension (fixed window + neg)
    for dim in embed_dims:
        model, losses = train_skipgram(vocab, e_dim=dim, window_size=3, num_neg=5, epochs=EXPERIMENT_EPOCHS, lr=0.005)
        name = f"sg_dim{dim}_win3_neg5"
        save_model(model, vocab, name)
        results["skipgram"].append({"name": name, "dim": dim, "window": 3, "neg": 5, "final_loss": losses[-1], "losses": losses})

    # stage 4: vary context window
    for win in window_sizes:
        model, losses = train_skipgram(vocab, e_dim=100, window_size=win, num_neg=5, epochs=EXPERIMENT_EPOCHS, lr=0.005)
        name = f"sg_dim100_win{win}_neg5"
        save_model(model, vocab, name)
        results["skipgram"].append({"name": name, "dim": 100, "window": win, "neg": 5, "final_loss": losses[-1], "losses": losses})

    # stage 5: vary number of negative samples
    for neg in neg_samples:
        model, losses = train_skipgram(vocab, e_dim=100, window_size=3, num_neg=neg, epochs=EXPERIMENT_EPOCHS, lr=0.005)
        name = f"sg_dim100_win3_neg{neg}"
        save_model(model, vocab, name)
        results["skipgram"].append({"name": name, "dim": 100, "window": 3, "neg": neg, "final_loss": losses[-1], "losses": losses})

    results_serializable = {
        k: [
            {
                "name": r["name"],
                "dim": r.get("dim"),
                "window": r.get("window"),
                "neg": r.get("neg"),
                "final_loss": r["final_loss"],
            }
            for r in v
        ]
        for k, v in results.items()
    }
    with open(os.path.join(O_DIR, "experiment_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2)

    plot_training_curves(results)

    print("all experiments complete!")
    print("=" * 70)
    print_results_table(results)

    return results, vocab


def save_model(model, vocab, name):
    model_dir = os.path.join(O_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    embeddings = model.get_embeddings()
    np.save(os.path.join(model_dir, f"{name}_embeddings.npy"), embeddings)

    with open(os.path.join(model_dir, f"{name}_vocab.json"), 'w', encoding='utf-8') as f:
        json.dump(
            {"word2idx": vocab.word2idx, "idx2word": {str(k): v for k, v in vocab.idx2word.items()}},
            f,
        )

    with open(os.path.join(model_dir, f"{name}_model.pth"), 'wb') as f:
        # .pth extension is preserved for compatibility with existing naming convention.
        # Failure point: this is a pickle payload, not a torch.load-compatible state_dict.
        # keeping this name so your existing report/results paths stay unchanged
        pickle.dump(model.state_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  model saved: {name}")


def plot_training_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for r in results["cbow"]:
        axes[0].plot(r["losses"], label=r["name"])
    axes[0].set_title("CBOW Training Loss", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

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
    print("training curves saved.")


def print_results_table(results):
    print(f"\n{'model':<30s} {'dim':>5s} {'win':>5s} {'neg':>5s} {'final loss':>12s}")
    print("-" * 62)
    for m_typ in ["cbow", "skipgram"]:
        for r in results[m_typ]:
            neg = r.get("neg", "-")
            print(
                f"{r['name']:<30s} {r.get('dim', ''):>5} {r.get('window', ''):>5} "
                f"{str(neg):>5s} {r['final_loss']:>12.4f}"
            )


if __name__ == "__main__":
    results, vocab = run_experiments()
