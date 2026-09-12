import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTM, self).__init__()
        self.l_it = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.l_c_t = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.l_ft = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.l_ot = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, c, h, bi, bf, bc, bo):
        ft = torch.sigmoid(self.l_ft(h) + bf)
        it = torch.sigmoid(self.l_it(h) + bi)
        c_t = torch.tanh(self.l_c_t(h) + bc)
        ct = c * ft + it * c_t
        ot = torch.sigmoid(self.l_ot(h) + bo)
        ht = ot * torch.tanh(ct)
        return ct, ht


class LSTMMultiUpdateBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTMMultiUpdateBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_blocks = LSTM(hidden_dim)

    def forward(self, net_c, net_h, inp):
        net_c, net_h = self.lstm_blocks(net_c, net_h, *inp)
        return net_c, net_h


class IterativeUnit(nn.Module):
    def __init__(self, hidden_dim, iters):
        super(IterativeUnit, self).__init__()
        self.hidden_dim = hidden_dim
        self.iters = iters
        self.update_block = LSTMMultiUpdateBlock(self.hidden_dim)
        self.bias = nn.Linear(self.hidden_dim, self.hidden_dim * 4)

    def forward(self, inputs):
        net_h = torch.tanh(inputs)
        net_ext = torch.relu(sum(inputs))
        net_ext = list(self.bias(net_ext).split(split_size=self.bias.out_features // 4, dim=-1))
        net_c = net_h
        for itr in range(self.iters):
            net_c, net_h = self.update_block(net_c, net_h, net_ext)
        return net_c
