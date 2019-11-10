import torch
import numpy as numpy

from torch import nn

class Encoder(nn.Module):

    def __init__(self, inp_dim, hid_dim, enc_dim, batch_size, n_layers):
        self.input_dim = inp_dim
        self.hidden_dim = hid_dim
        self.encoder_dim = enc_dim
        self.batch_size = batch_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            self.input_dim, 
            self.hidden_state,
            self.n_layers)
            
        self.out_linear = nn.Linear(self.hidden_dim, self.encoder_dim)

    def forward(self):
        pass


class Decoder(nn.Module):

    def __init__(self, inp_dim, hid_dim, enc_dim, batch_size):
        self.input_dim = inp_dim
        self.hidden_dim = hid_dim
        self.decoder_dim = enc_dim
        self.batch_size = batch_size

    def forward(self):
        pass


class Sequence2Sequence:

    def __init__(self, enc, dec):
        self.encoder = enc
        self.decoder = dec

    def forward(self):
        pass