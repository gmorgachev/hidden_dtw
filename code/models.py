import torch
import numpy as numpy

from torch import nn

class Encoder(nn.Module):

    def __init__(self, inp_dim, hid_dim, enc_dim, batch_size, n_layers):
        super(Encoder, self).__init__()
        self.input_dim = inp_dim
        self.hidden_dim = hid_dim
        self.encoder_dim = enc_dim
        self.batch_size = batch_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            self.input_dim, 
            self.hidden_dim,
            self.n_layers)
            
        # self.out_linear = nn.Linear(self.hidden_dim, self.encoder_dim)

    def forward(self, input, h_0, c_0):
        """
        input: length x batch_size x input_dim
        """
        out, (h_n, c_n) = self.lstm(input, (h_0, c_0))

        return out, (h_n, c_n)

class Decoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, enc_dim, batch_size, n_layers):
        super(Decoder, self).__init__()
        self.input_dim = inp_dim
        self.hidden_dim = hid_dim
        self.decoder_dim = enc_dim
        self.batch_size = batch_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            self.input_dim, 
            self.hidden_dim,
            self.n_layers)

    def forward(self, input, h_n, c_n):
        """
        input: length x batch_size x input_dim
        """
        out, (h_n, c_n) = self.lstm(input, (h_n, c_n))

        return out, (h_n, c_n)


class Sequence2Sequence(nn.Module):

    def __init__(self, enc, dec):
        super(Sequence2Sequence, self).__init__()
        self.encoder = enc
        self.decoder = dec

    def forward(self, input, h_enc, c_enc, h_dec, c_dec):
        encoded, (h, c) = self.encoder(input, h_enc, c_enc)
        out, _ = self.decoder(encoded, h_dec, c_dec)

        return out