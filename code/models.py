import torch
import numpy as numpy

from torch import nn

class Encoder(nn.Module):

    def __init__(self, inp_dim, hid_dim, enc_dim, n_layers, bidirectional=False):
        super(Encoder, self).__init__()
        self.input_dim = inp_dim
        self.hidden_dim = hid_dim
        self.encoder_dim = enc_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.n_layers,
            bidirectional=bidirectional)
            
        # self.out_linear = nn.Linear(self.hidden_dim, self.encoder_dim)
    
    def forward(self, input):
        """
        input: length x batch_size x input_dim
        """
        h_n = self.init_hidden(input)
        c_n = self.init_hidden(input)
        out, (h_n, c_n) = self.lstm(input, (h_n, c_n))

        return out, (h_n, c_n)

    def init_hidden(self, x):
        return torch.zeros((1+self.lstm.bidirectional) * self.n_layers,
            x.shape[1], self.hidden_dim).to(x.device)

class Decoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, enc_dim, n_layers):
        super(Decoder, self).__init__()
        self.input_dim = inp_dim
        self.hidden_dim = hid_dim
        self.decoder_dim = enc_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.n_layers)

    def forward(self, input):
        """
        input: length x batch_size x input_dim
        """
        h_n = self.init_hidden(input)
        c_n = self.init_hidden(input)
        out, (h_n, c_n) = self.lstm(input, (h_n, c_n))

        return out, (h_n, c_n)

    def init_hidden(self, x):
        return torch.zeros(self.n_layers, x.shape[1], self.hidden_dim).to(x.device)


class Sequence2Sequence(nn.Module):

    def __init__(self, enc, dec):
        super(Sequence2Sequence, self).__init__()
        self.encoder = enc
        self.decoder = dec

    def forward(self, input):
        encoded, (h, c) = self.encoder(input)
        encoded = encoded.view(input.shape[0], input.shape[1],
            1+self.encoder.lstm.bidirectional, self.encoder.hidden_dim)[:, :, 0, :]
        out, _ = self.decoder(encoded)

        return out
    
    
class Autoencoder(nn.Module):
    def __init__(self, batch_size, seq_len, in_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers)
        self.linear_layer = nn.Linear(hidden_dim, in_dim)
        
    def forward(self, inp):
        h0, c0 = self.get_init_state(inp.shape[1])
        
        out, _ = self.lstm(inp, (h0, c0))
        out = self.linear_layer(out)
        
        return out
        
    def get_init_state(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim),\
               torch.zeros(1, batch_size, self.hidden_dim)