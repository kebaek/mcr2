import torch
import torch.nn as nn
import torch.nn.functional as F


# class MLP(nn.Module):
#     def __init__(self, in_channels=3, hidden_channels=64, num_layers=3,
#                  dropout=0, use_bn=False):
#         super(MLP, self).__init__()
#         self.lins = nn.Sequential(
#             nn.Linear(3, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Linear(32, 32),
#             # nn.ReLU()
#         )

#     def forward(self, x):    
#         x = self.lins(x)
#         x = F.normalize(x)
#         return x


class MLP(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, num_layers=3,
                 dropout=0, use_bn=False):
        super(MLP, self).__init__()
        self.lins = nn.Sequential(
            # nn.Linear(3, 16),
            # nn.BatchNorm1d(16),
            # nn.ReLU(),
            # nn.Linear(16, 16),
            nn.Linear(3, 20),
            # nn.BatchNorm1d(3),
            nn.ReLU(),
            nn.Linear(20, 20),
            # nn.ReLU()
        )

    def forward(self, x):    
        x = self.lins(x)
        x = F.normalize(x)
        return x


# class MLP(nn.Module):
#     def __init__(self, in_channels=3, hidden_channels=64, num_layers=3,
#                  dropout=0, use_bn=False):
#         super(MLP, self).__init__()
#         self.lins = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         if num_layers == 1:
#             # just linear layer i.e. logistic regression
#             self.lins.append(nn.Linear(in_channels, out_channels))
#         else:
#             self.lins.append(nn.Linear(in_channels, hidden_channels))
#             self.bns.append(nn.BatchNorm1d(hidden_channels))
#             for _ in range(num_layers - 2):
#                 self.lins.append(nn.Linear(hidden_channels, hidden_channels))
#                 self.bns.append(nn.BatchNorm1d(hidden_channels))
#             self.lins.append(nn.Linear(hidden_channels, hidden_channels))

#         self.dropout = dropout
#         self.use_bn = use_bn

#     def reset_parameters(self):
#         for lin in self.lins:
#             lin.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()

#     def forward(self, x):    
#         for i, lin in enumerate(self.lins[:-1]):
#             x = lin(x)
#             x = F.relu(x, inplace=True)
#             if self.use_bn:
#                 x = self.bns[i](x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.lins[-1](x)
#         x = F.normalize(x)
#         return x
