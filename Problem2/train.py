# train all three models on same split/settings
# kept in one file so comparison is easy after training

# assignment 2
# couldn't get the other way to work so did it like this
import os
# assignment 2
# couldn't get the other way to work so did it like this
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# add parent to path for model imports
sys.path.insert(0, os.path.dirname(__file__))
from models import CVocab, VRNN, BLSTM, RNNAttn

D_PATH = os.path.join(os.path.dirname(__file__), "TrainingNames.txt")
O_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(O_DIR, exist_ok=True)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NDset(Dataset):
    # simple shift-by-one dset
    # input  = seq[:-1]
    # target = seq[1:]

    def __init__(self, names, vocab, m_len=None):
        self.vocab = vocab
        # encode all names
        self.encoded = [vocab.encode(name) for name in names]
        # find maximum length
        if m_len is None:
            m_len = max(len(e) for e in self.encoded)
        self.m_len = m_len

        # pad all sequences to m_len
        self.data = []
        for e in self.encoded:
            # pad with pad tokens to reach m_len
            padded = e + [vocab.pad_idx] * (m_len - len(e))
            # truncate if longer than m_len
            self.data.append(padded[:m_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # input: all but last token
        # target: all but first token (shifted by 1)
        # this creates input-output pairs for training
        return (torch.tensor(seq[:-1], dtype=torch.long),
                torch.tensor(seq[1:], dtype=torch.long))

def ld_nms():
    
    names = []
    with open(D_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            # keep non-empty names of reasonable length
            if name and len(name) >= 2:
                names.append(name)
    print(f"loaded {len(names)} names.")
    return names

    # trying a different way because the original was too slow
def tr_m(model, dl, vocab, m_nm,
                epochs=100, lr=0.003, c_grd=5.0):
    # train one model config
    # c_grd is important here, without it rnn sometimes explodes
    
    print(f"training: {m_nm}")
    print(f"parameters: {model.count_parameters():,}")
    print(f"device: {DEV}")
    

    model = model.to(DEV)
    # adam opt with default betas
    opt = optim.Adam(model.parameters(), lr=lr)
    # reduce learning rate if loss plateaus (helps convergence)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    # cross-entropy loss, ignore pad tokens in loss calculation
    crit = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    losses = []
    b_loss = float('inf')  # seems to work

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        n_bs = 0
        start = time.time()

    # trying a different way because the original was too slow
        for ins, tgts in dl:
            ins, tgts = ins.to(DEV), tgts.to(DEV)

            # forward pass
            opt.zero_grad()
            logits, _ = model(ins)  # (batch, seq_len, v_sz)

            # loss computation
            # reshape: (batch*seq_len, v_sz) vs (batch*seq_len,)
            loss = crit(logits.reshape(-1, vocab.v_sz), tgts.reshape(-1))
            loss.backward()

            # tried without clip once; loss became unstable
            torch.nn.utils.clip_grad_norm_(model.parameters(), c_grd)
            opt.step()

            t_loss += loss.item()
            n_bs += 1

        # average loss for this epoch
        a_loss = t_loss / max(n_bs, 1)
        losses.append(a_loss)
        # adjust learning rate based on loss
        sch.step(a_loss)
        elapsed = time.time() - start

        # print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {a_loss:.4f} | "
                  f"LR: {opt.param_groups[0]['lr']:.6f} | Time: {elapsed:.1f}s")

        # save best model (lowest loss)
        if a_loss < b_loss:
            b_loss = a_loss
            torch.save({
                'model_state_dict': model.state_dict(),  # checking this
                'epoch': epoch,
                'loss': b_loss,
            }, os.path.join(O_DIR, f"{m_nm}_best.pth"))

    return losses

def pl_loss(a_ls, save_path):
    
    plt.figure(figsize=(12, 6))
    # tried batch size 128 but memory exploded
    for name, losses in a_ls.items():
        plt.plot(losses, label=name, linewidth=2)
    plt.title("Training Loss - Character-Level Name Generation", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Loss", fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # checking this
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"loss plot saved to: {save_path}")

def main():
    # load data
    names = ld_nms()
    vocab = CVocab(names)  # seems to work
    print(f"character voc size: {vocab.v_sz}")
    print(f"characters: {vocab.chars}")

    # save vocab
    v_dat = {
        'char2idx': vocab.char2idx,
        'idx2char': {str(k): v for k, v in vocab.idx2char.items()},
        'v_sz': vocab.v_sz,
    }
    with open(os.path.join(O_DIR, "char_vocab.json"), 'w') as f:
        json.dump(v_dat, f, indent=2)

    # create dset
    dset = NDset(names, vocab)
    dl = DataLoader(dset, batch_size=64, shuffle=True, num_workers=0)

    # h_prm
    e_dim = 32
    h_sz = 128
    n_lyrs = 1
    lr = 0.003
    epochs = 100

    h_prm = {
        "e_dim": e_dim,
        "h_sz": h_sz,
        "n_lyrs": n_lyrs,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": 64,
        "c_grd": 5.0,
        "opt": "Adam",
        "sch": "ReduceLROnPlateau",
    }

    # train all 3 models
    m_cfg = {
        "vanilla_rnn": VRNN(vocab.v_sz, e_dim, h_sz, n_lyrs),
        "blstm": BLSTM(vocab.v_sz, e_dim, h_sz, n_lyrs),
        "rnn_attention": RNNAttn(vocab.v_sz, e_dim, h_sz, n_lyrs),
    }

    a_ls = {}
    m_inf = {}

    for name, model in m_cfg.items():
        n_params = model.count_parameters()
        print(f"\n{name}: {n_params:,} trainable parameters")

        losses = tr_m(model, dl, vocab, name, epochs=epochs, lr=lr)
        a_ls[name] = losses

        m_inf[name] = {
            "trainable_params": n_params,
            "final_loss": losses[-1],
            "b_loss": min(losses),
            "h_prm": h_prm,
        }

    # save training info
    with open(os.path.join(O_DIR, "training_info.json"), 'w') as f:
        json.dump(m_inf, f, indent=2)

    # plot loss curves
    pl_loss(a_ls, os.path.join(O_DIR, "training_loss.png"))

    
    print("training complete!")
    
    # idk why but this fixes the convergence issue
    for name, info in m_inf.items():
        print(f"  {name:20s} | Params: {info['trainable_params']:>8,} | "
              f"Best Loss: {info['b_loss']:.4f}")  # checking this

if __name__ == "__main__":
    main()
