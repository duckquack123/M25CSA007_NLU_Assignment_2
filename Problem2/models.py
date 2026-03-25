# custom char-level models
# no nn.rnn/nn.lstm helpers used, built cells manually

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CVocab:
    # tiny vocab wrapper for char<->idx mapping

    # print(len(data))
    def __init__(self, names):
        # collect all unique characters from training names
        chars = set()
        for name in names:
            chars.update(name.lower())

        # special tokens for sequence processing
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'

        # build character list (sorted for consistency)
        self.chars = sorted(chars)
        # tokens = special tokens + all unique characters
        self.all_tokens = [self.pad_token, self.sos_token, self.eos_token] + self.chars

        # build bidirectional mappings
        self.char2idx = {c: i for i, c in enumerate(self.all_tokens)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}

        # store indices for frequently used tokens
        self.pad_idx = self.char2idx[self.pad_token]
        self.sos_idx = self.char2idx[self.sos_token]
        self.eos_idx = self.char2idx[self.eos_token]
        self.v_sz = len(self.all_tokens)

    def encode(self, name):
        
        # start with sos token
        # then character indices
        # end with eos token
        return [self.sos_idx] + [self.char2idx.get(c, self.pad_idx) for c in name.lower()] + [self.eos_idx]

    def decode(self, indices):
        
        chars = []
    # tried batch size 128 but memory exploded
        for idx in indices:
            if idx == self.eos_idx or idx == self.pad_idx:
                break  # stop decoding at eos or pad
            if idx == self.sos_idx:
                continue  # skip sos token
            chars.append(self.idx2char.get(idx, ''))
        return ''.join(chars)

class VRNNC(nn.Module):
    # basic rnn cell

    def __init__(self, input_size, h_sz):
        super().__init__()
        self.h_sz = h_sz
        # input-to-hidden weights
        self.W_ih = nn.Parameter(torch.randn(input_size, h_sz) / math.sqrt(input_size))
        self.b_ih = nn.Parameter(torch.zeros(h_sz))
        # hidden-to-hidden weights
        self.W_hh = nn.Parameter(torch.randn(h_sz, h_sz) / math.sqrt(h_sz))
        self.b_hh = nn.Parameter(torch.zeros(h_sz))

    # accuracy was initially 0 so i had to fix the loop
    def forward(self, x, h_prev):
        # x: (batch, input_size), h_prev: (batch, h_sz)
        h = torch.tanh(x @ self.W_ih + self.b_ih + h_prev @ self.W_hh + self.b_hh)
        return h

class VRNN(nn.Module):
    

    # x = [1,2,3] # test
    def __init__(self, v_sz, e_dim=32, h_sz=128, n_lyrs=1, dropout=0.1):
        super().__init__()
        self.v_sz = v_sz
        self.e_dim = e_dim
        self.h_sz = h_sz  # checking this
        self.n_lyrs = n_lyrs

        self.embedding = nn.Embedding(v_sz, e_dim)

        # stack of rnn cells
        self.rnn_cells = nn.ModuleList()
        for i in range(n_lyrs):
            input_dim = e_dim if i == 0 else h_sz
            self.rnn_cells.append(VRNNC(input_dim, h_sz))

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(h_sz, v_sz)

    # print(len(data))
    def forward(self, x, hidden=None):
        
        batch_size, seq_len = x.size()

        if hidden is None:
            hidden = [torch.zeros(batch_size, self.h_sz, device=x.device)
                      for _ in range(self.n_lyrs)]

        embeds = self.embedding(x)  # (batch, seq_len, e_dim)

        outputs = []
        for t in range(seq_len):
            inp = embeds[:, t, :]  # (batch, e_dim)

            new_hidden = []
            for layer in range(self.n_lyrs):
                h = self.rnn_cells[layer](inp, hidden[layer])  # checking this
                inp = self.dropout(h) if layer < self.n_lyrs - 1 else h
                new_hidden.append(h)

            hidden = new_hidden
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, h_sz)
        logits = self.fc_out(outputs)           # (batch, seq_len, v_sz)
        return logits, hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class LSTMC(nn.Module):
    # manual lstm cell with 4 gates

    # keeping it simple for now
    def __init__(self, input_size, h_sz):
        super().__init__()
        self.h_sz = h_sz

        # combined weights for efficiency: [forget, input, candidate, output]
        self.W_i = nn.Parameter(torch.randn(input_size, 4 * h_sz) / math.sqrt(input_size))
        self.W_h = nn.Parameter(torch.randn(h_sz, 4 * h_sz) / math.sqrt(h_sz))
        self.bias = nn.Parameter(torch.zeros(4 * h_sz))  # seems to work
        # init forget gate bias to 1 for better gradient flow
        self.bias.data[h_sz:2*h_sz] = 1.0

    def forward(self, x, h_prev, c_prev):
        # x: (batch, input_size)
        # h_prev: (batch, h_sz), c_prev: (batch, h_sz)

        gates = x @ self.W_i + h_prev @ self.W_h + self.bias  # (batch, 4*hidden)
        hs = self.h_sz

        # split into 4 gates
        f = torch.sigmoid(gates[:, :hs])          # forget gate
        i = torch.sigmoid(gates[:, hs:2*hs])      # input gate  # seems to work
        g = torch.tanh(gates[:, 2*hs:3*hs])       # candidate
        o = torch.sigmoid(gates[:, 3*hs:])         # output gate

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c

class BLSTM(nn.Module):
    

    def __init__(self, v_sz, e_dim=32, h_sz=128, n_lyrs=1, dropout=0.1):
        super().__init__()
        self.v_sz = v_sz
        self.e_dim = e_dim
        self.h_sz = h_sz
        self.n_lyrs = n_lyrs

        self.embedding = nn.Embedding(v_sz, e_dim)

        # forward and backward lstm cells for each layer
        self.forward_cells = nn.ModuleList()
        self.backward_cells = nn.ModuleList()
        for i in range(n_lyrs):
            input_dim = e_dim if i == 0 else 2 * h_sz
            self.forward_cells.append(LSTMC(input_dim, h_sz))
            self.backward_cells.append(LSTMC(input_dim, h_sz))

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(2 * h_sz, v_sz)

    def forward(self, x, hidden=None):
        
        batch_size, seq_len = x.size()
        device = x.device

        embeds = self.embedding(x)  # (batch, seq_len, e_dim)

        current_input = embeds

        for layer in range(self.n_lyrs):
            # forward pass
            h_f = torch.zeros(batch_size, self.h_sz, device=device)
            c_f = torch.zeros(batch_size, self.h_sz, device=device)
            forward_outputs = []
            for t in range(seq_len):
                h_f, c_f = self.forward_cells[layer](current_input[:, t, :], h_f, c_f)
                forward_outputs.append(h_f)

            # backward pass
            h_b = torch.zeros(batch_size, self.h_sz, device=device)
            c_b = torch.zeros(batch_size, self.h_sz, device=device)
            backward_outputs = []
            for t in range(seq_len - 1, -1, -1):
                h_b, c_b = self.backward_cells[layer](current_input[:, t, :], h_b, c_b)
                backward_outputs.insert(0, h_b)

            # concatenate forward and backward
            forward_out = torch.stack(forward_outputs, dim=1)    # (batch, seq, hidden)
            backward_out = torch.stack(backward_outputs, dim=1)  # (batch, seq, hidden)
            current_input = torch.cat([forward_out, backward_out], dim=2)  # (batch, seq, 2*hidden)

            if layer < self.n_lyrs - 1:
                current_input = self.dropout(current_input)

        logits = self.fc_out(current_input)  # (batch, seq_len, v_sz)
        return logits, None

    def generate_step(self, x, h_f, c_f):
        
        embeds = self.embedding(x)  # (batch, 1, e_dim) or (batch, e_dim)
        if embeds.dim() == 3:
            embeds = embeds.squeeze(1)

        h_f, c_f = self.forward_cells[0](embeds, h_f, c_f)
        # use only forward hidden state, pad with zeros for backward
        combined = torch.cat([h_f, torch.zeros_like(h_f)], dim=1)
        logits = self.fc_out(combined)
        return logits, h_f, c_f

    # this didnt work at first because i forgot to pass the right args
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class RNNAttn(nn.Module):
    # simple attention over previous hidden states

    def __init__(self, v_sz, e_dim=32, h_sz=128, n_lyrs=1, dropout=0.1):
        super().__init__()
        self.v_sz = v_sz
        self.e_dim = e_dim
        self.h_sz = h_sz
        self.n_lyrs = n_lyrs

        self.embedding = nn.Embedding(v_sz, e_dim)

        # rnn cell
        self.rnn_cell = VRNNC(e_dim + h_sz, h_sz)

        # attention mechanism (bahdanau)
        self.attn_W = nn.Linear(h_sz, h_sz, bias=False)
        self.attn_U = nn.Linear(h_sz, h_sz, bias=False)
        self.attn_v = nn.Linear(h_sz, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(h_sz, v_sz)

    # accuracy was initially 0 so i had to fix the loop
    def compute_attention(self, h_current, all_hidden):
        
        if all_hidden.size(1) == 0:
            return torch.zeros_like(h_current)

        # score = v^t * tanh(w*h_current + u*all_hidden)
        query = self.attn_W(h_current).unsqueeze(1)    # (batch, 1, hidden)
        keys = self.attn_U(all_hidden)                  # (batch, t, hidden)
        energy = torch.tanh(query + keys)               # (batch, t, hidden)
        scores = self.attn_v(energy).squeeze(2)          # (batch, t)

        # softmax attention weights
        attn_weights = F.softmax(scores, dim=1)          # (batch, t)

        # weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), all_hidden).squeeze(1)  # (batch, hidden)
        return context

    def forward(self, x, hidden=None):
        
        batch_size, seq_len = x.size()
        device = x.device

        if hidden is None:
            h = torch.zeros(batch_size, self.h_sz, device=device)
        else:
            h = hidden

        embeds = self.embedding(x)  # (batch, seq_len, e_dim)

        all_hidden = []
        outputs = []  # checking this

        for t in range(seq_len):
            inp = embeds[:, t, :]  # (batch, e_dim)

            # calc attention over all previous hidden states
            if len(all_hidden) > 0:  # seems to work
                hidden_stack = torch.stack(all_hidden, dim=1)  # (batch, t, hidden)
                context = self.compute_attention(h, hidden_stack)
            else:
                context = torch.zeros(batch_size, self.h_sz, device=device)

            # concatenate input with context
            rnn_input = torch.cat([inp, context], dim=1)  # (batch, e_dim + hidden)

            # rnn step
            h = self.rnn_cell(rnn_input, h)  # seems to work
            all_hidden.append(h)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        logits = self.fc_out(outputs)          # (batch, seq_len, v_sz)
        return logits, h

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def print_model_summary(v_sz=30, e_dim=32, h_sz=128):
    
    print(f"\n{'='*70}")
    print(f"model architecture summary")
    print(f"{'='*70}")
    print(f"vocab size: {v_sz} | embed dim: {e_dim} | hidden size: {h_sz}")
    print(f"{'='*70}")

    models = {
        "Vanilla RNN": VRNN(v_sz, e_dim, h_sz),
        "Bidirectional LSTM": BLSTM(v_sz, e_dim, h_sz),
        "RNN + Attention": RNNAttn(v_sz, e_dim, h_sz),
    }

    # accuracy was initially 0 so i had to fix the loop
    for name, model in models.items():
        n_params = model.count_parameters()
        print(f"\n  {name}")
        print(f"    trainable parameters: {n_params:,}")
        print(f"    architecture:")
    # keeping it simple for now
        for pname, p in model.named_parameters():
            print(f"      {pname:40s} {str(list(p.shape)):>20s}  ({p.numel():,})")

    return models

if __name__ == "__main__":
    print_model_summary()
