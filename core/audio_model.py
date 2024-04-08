import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        attention_weights = F.softmax(self.fc(hidden_states).squeeze(2), dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(2) * hidden_states, dim=1)
        return context_vector, attention_weights

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)


class RecurrentAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.3):
        super(RecurrentAttentionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.attention1 = Attention(hidden_size, dropout_rate)
        self.attention2 = Attention(hidden_size, dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def forward(self, x):
        output, hn = self.gru(x)
        context_vector1, attention_weights1 = self.attention1(output)
        context_vector2, attention_weights2 = self.attention2(output)
        context_vector = torch.cat([context_vector1, context_vector2], dim=1)
        context_vector = self.dropout(context_vector)
        context_vector = self.batch_norm(context_vector)
        out = torch.relu(self.fc(context_vector))
        return out.squeeze(1)

    def init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                param.data.fill_(0)
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='relu')
        nn.init.constant_(self.fc.bias, 0)
