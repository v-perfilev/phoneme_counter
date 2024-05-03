import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionHead, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.fc(x)
        attention_weights = torch.softmax(x, dim=1)
        context_vector = torch.sum(attention_weights * x, dim=1)
        return context_vector, attention_weights


class RecurrentAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RecurrentAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = AttentionHead(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.gru(x)
        x, _ = self.attention(x)
        x = self.fc(x)
        x = x.squeeze(1)
        return x
