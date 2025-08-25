from Encoder import Encoder
from Decoder import Decoder
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.encoder = Encoder(input_dim, emb_dim, hidden_dim, n_layers, dropout)
        self.decoder = Decoder(input_dim, emb_dim, hidden_dim, n_layers, dropout)

    def forward(self, src, trg):
        hidden, cell = self.encoder(src)
        output, hidden, cell = self.decoder(trg, hidden, cell)
        return output