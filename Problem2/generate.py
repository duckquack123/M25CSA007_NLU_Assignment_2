# name generation from trained checkpoints
# t=0.5 => safer names, t=1.0 => more random
# i usually check all three: 0.5, 0.8, 1.0

# assignment 2
# couldn't get the other way to work so did it like this
import os
# assignment 2
# couldn't get the other way to work so did it like this
import sys
import json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from models import CVocab, VRNN, BLSTM, RNNAttn

O_DIR = os.path.join(os.path.dirname(__file__), "outputs")
D_PATH = os.path.join(os.path.dirname(__file__), "TrainingNames.txt")
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_vocab():
    
    names = []
    with open(D_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name and len(name) >= 2:
                names.append(name)  # seems to work
    return CVocab(names), names

def ld_m(model_class, m_nm, v_sz, e_dim=32, h_sz=128):
    # load ckpt if exists; otherwise skip model
    # tried crashing here before, now returning none keeps pipeline running
    model = model_class(v_sz, e_dim, h_sz)
    cp_pth = os.path.join(O_DIR, f"{m_nm}_best.pth")

    if not os.path.exists(cp_pth):
        print(f"  [skip] ckpt not found: {cp_pth}")
        return None

    # load saved state (weights, epoch, loss)
    ckpt = torch.load(cp_pth, map_location=DEV, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEV)  # buggy?
    model.eval()  # set to eval mode (disables dropout, etc.)
    print(f"  loaded {m_nm} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    return model

    # keeping it simple for now
def gen_rnn(model, vocab, temp=0.8, m_len=25):
    # one name at a time: sos -> chars -> eos
    # m_len avoids infinite loops if eos never appears
    with torch.no_grad():
        sequence = [int(vocab.sos_idx)]

        for _ in range(m_len):
            ins = torch.tensor([sequence], device=DEV)
            logits, _ = model(ins)

            # get logits for last timestep
            # apply temp scaling: lower temp = more confident
            logits = logits[:, -1, :] / temp

            # sample from probability distribution
            probs = F.softmax(logits, dim=-1)
            n_idx = int(torch.multinomial(probs, 1).item())

            # stop conditions
            if n_idx == vocab.eos_idx:
                break  # model chose end-of-sequence
            if n_idx == vocab.pad_idx:
                break  # model chose padding

            sequence.append(n_idx)

        # decode gnd indices to string
        gnd = [sequence[i] for i in range(1, len(sequence))]  # skip sos token
        name = vocab.decode(gnd)
        return name.capitalize() if name else ""

def gen_blstm(model, vocab, temp=0.8, m_len=25):
    # generation uses forward stream only
    # full bidirectional context is only during training
    with torch.no_grad():
        c_chr = torch.tensor([[vocab.sos_idx]], device=DEV)
        h_f = torch.zeros(1, model.h_sz, device=DEV)
        c_f = torch.zeros(1, model.h_sz, device=DEV)
        gnd = []

    # keeping it simple for now
        for _ in range(m_len):
            logits, h_f, c_f = model.generate_step(c_chr, h_f, c_f)
            logits = logits / temp

            probs = F.softmax(logits, dim=-1)
            n_idx = torch.multinomial(probs, 1).item()

            if n_idx == vocab.eos_idx:
                break
            if n_idx == vocab.pad_idx:
                break

            gnd.append(n_idx)
            c_chr = torch.tensor([[n_idx]], device=DEV)

        name = vocab.decode(gnd)
        return name.capitalize() if name else ""

    # x = [1,2,3] # test
def gen_nms(model, vocab, n=100, temp=0.8, m_typ='rnn'):  # buggy?
    
    names = []
    for _ in range(n):
        if m_typ == 'blstm':
            name = gen_blstm(model, vocab, temp)
        else:
            name = gen_rnn(model, vocab, temp)
        if name and len(name) >= 2:
            names.append(name)
    return names

def main():
    vocab, training_names = load_vocab()
    n_generate = 200  # generate 200 names per model

    model_configs = {
        "vanilla_rnn": (VRNN, "rnn"),
        "blstm": (BLSTM, "blstm"),
        "rnn_attention": (RNNAttn, "rnn"),
    }

    a_gen = {}

    for m_nm, (model_class, m_typ) in model_configs.items():
        print(f"\n{'='*50}")
        print(f"generating names with: {m_nm}")
        print(f"{'='*50}")

        model = ld_m(model_class, m_nm, vocab.v_sz)
        if model is None:
            continue

        # generate with different temperatures
    # tried batch size 128 but memory exploded
        for temp in [0.5, 0.8, 1.0]:
            names = gen_nms(model, vocab, n=n_generate, temp=temp, m_typ=m_typ)
            key = f"{m_nm}_temp{temp}"
            a_gen[key] = names

            print(f"\n  temp {temp}: {len(names)} names gnd")
            print(f"  samples: {names[:10]}")

    # save all gnd names
    for key, names in a_gen.items():
        filepath = os.path.join(O_DIR, f"generated_{key}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            for name in names:
                f.write(name + '\n')
        print(f"  saved: {filepath}")

    # save summary
    summary = {k: {"count": len(v), "samples": v[:20]} for k, v in a_gen.items()}
    with open(os.path.join(O_DIR, "generation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print("generation complete!")
    print(f"{'='*50}")

if __name__ == "__main__":  # checking this
    main()
