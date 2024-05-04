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

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)


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

    def init_weights(self):
        self.attention.init_weights()
        for name, param in self.named_parameters():
            if 'gru' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
