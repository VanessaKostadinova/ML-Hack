from random import Random

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, emb_model, h_size, n_layers, dropout=0.5, device="cpu"):
        super(Encoder, self).__init__()

        self.hidden_size = h_size
        self.num_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.emb_model = emb_model
        self.device = device

        embedding_weights = torch.tensor(self.emb_model.vectors, device=self.device)
        self.emb_size = embedding_weights.size()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)

        self.rnn = nn.LSTM(list(self.emb_size)[1], h_size, n_layers, dropout=dropout, batch_first=True)

    def forward(self, word_in, h_0=None):
        try:
            # try getting embedding
            embedding = torch.tensor(self.emb_model[word_in], device=self.device)
        except KeyError:
            # if vector unknown: set to zeroes
            embedding = torch.zeros(size=[1, list(self.emb_size)[1]], device=self.device)

        emb_in = self.dropout(embedding)

        if h_0 is None:
            out, h_out = self.rnn(emb_in.unsqueeze(0))
        else:
            out, h_out = self.rnn(emb_in.unsqueeze(0), h_0)

        return out, h_out


class Decoder(nn.Module):
    def __init__(self, h_size):
        super().__init__()
        self.fc_0 = nn.Linear(h_size, 300)
        self.fc_1 = nn.Linear(300, 200)
        self.fc_out = nn.Linear(200, 1)

    def forward(self, ctx_vec):
        x = self.fc_0(ctx_vec)
        x = self.fc_1(x)
        prediction = self.fc_out(x)
        prediction = prediction.squeeze(0)
        prediction = prediction.to(torch.int64)
        return prediction


class EncoderFC(nn.Module):
    def __init__(self, encoder, decoder, h_dim, device):
        super(EncoderFC, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.h_dim = h_dim
        self.random = Random()

    def forward(self, tokens):
        input_words = tokens

        enc_o, enc_h = self.encoder(input_words[0])

        for index in range(1, len(input_words)):
            enc_o, enc_h = self.encoder(input_words[index], enc_h)

        prediction = self.decoder(enc_o)

        return prediction

    def evaluate(self, tokens):
        input_words = tokens

        enc_o, enc_h = self.encoder(input_words[0])

        for index in range(1, len(input_words)):
            enc_o, enc_h = self.encoder(input_words[index], enc_h)

        prediction = self.decoder(enc_o)
        prediction = prediction.squeeze(0)
        return prediction
