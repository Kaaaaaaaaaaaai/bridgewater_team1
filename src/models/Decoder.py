import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        output = self.fc(output.squeeze(0))
        return output, hidden, cell