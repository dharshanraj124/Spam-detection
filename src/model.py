import torch
import torch.nn as nn

class SpamRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            num_layers=2, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        _, (h, _) = self.lstm(x)
        # Concatenate the final forward and backward hidden states
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        x = self.fc(h)
        return self.sig(x)
