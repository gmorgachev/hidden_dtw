import torch
import numpy as numpy

from torch import nn
    

class Encoder(nn.Module):
    def __init__(self, seq_len, in_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers)
        self.linear_layer = nn.Linear(hidden_dim, in_dim)
        
    def forward(self, inp):
        h_0, c_0 = self.get_init_state(inp)
        hiddens, (h_n, c_n) = self.lstm(inp, (h_0, c_0))
        
        return hiddens, (h_n, c_n)
        
    def get_init_state(self, inp):
        batch_size = inp.shape[1]
        return torch.zeros((1, batch_size, self.hidden_dim)).to(inp.device),\
               torch.zeros((1, batch_size, self.hidden_dim)).to(inp.device)


class DecoderLSTM(nn.Module):
    def __init__(self, seq_len, in_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(hidden_dim, in_dim, n_layers)
        self.linear_h = nn.Linear(hidden_dim, in_dim)
        self.linear_c = nn.Linear(hidden_dim, in_dim)
        
    def forward(self, inp, enc_state):
        h_0 = self.linear_h(enc_state[0])
        c_0 = self.linear_c(enc_state[1])
        
        out, _ = self.lstm(inp, (h_0, c_0))
        
        return out



class DecoderLinear(nn.Module):
    def __init__(self, seq_len, in_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.linear_layer = nn.Linear(hidden_dim, in_dim)
        
    def forward(self, inp, enc_state):
        return self.linear_layer(inp)


class Sequence2Sequence(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.encoder = enc
        self.decoder = dec

    def forward(self, inp):
        out = self.decoder(*self.encoder(inp))

        return out
    