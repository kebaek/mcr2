import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import functional as F
import load as L
from loss import MaximalCodingRateReduction
import metrics_sup
import plot2 as plot
import utils


class MCR2Variational(nn.Module):
    """Equation 9 in writeup. """
    def __init__(self, eps, mu):
        super(MCR2Variational, self).__init__()
        self.eps = eps
        self.mu = mu
        
    def loss_discrimn(self, Z):
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        return 0.5 * torch.logdet(I + d / (n * self.eps) * Z @ Z.T)

    def loss_compress(self, Z, Pi, Us):
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = 0.
        for j in range(Pi.shape[1]):
            trPi_j = Pi[:, j].sum()
            scalar_j = trPi_j / (2 * n)
            norms = torch.linalg.norm(Us[j], axis=0, keepdims=True, ord=2) ** 2
            compress_loss += scalar_j * torch.log(1 + d / (trPi_j * self.eps) * norms).sum()
        return compress_loss

    def reg_U(self, Z, Pi, Us):
        loss_reg = 0.
        for j in range(Us.shape[0]):
            norm = torch.linalg.norm((Z @ Pi[:, j].diag() @ Z.T) - (Us[j] @ Us[j].T), ord='fro')
            print('reg norm', j, norm)
            loss_reg += norm ** 2
        return 0.5 * self.mu * loss_reg
    
    def forward(self, Z, Pi, Us):
        loss_R = self.loss_discrimn(Z.T)
        loss_Rc = self.loss_compress(Z.T, Pi, Us)
        loss_reg_U = self.reg_U(Z.T, Pi, Us)
        loss_obj = loss_R - loss_Rc - loss_reg_U
        return -loss_obj, loss_R.item(), loss_Rc.item(), loss_reg_U.item()

def label_to_membership(labels):
    n_class = labels.max() + 1
    n_samples = labels.shape[0]
    membership = torch.zeros(n_samples, n_class)
    for j in range(n_class):
        idx_j = labels == j
        membership[idx_j, j] = 1.
    return membership


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='dataset for training')
parser.add_argument('--samples', type=int, required=True, help='number of samples from each class')
parser.add_argument('--arch', type=str, required=True, help='architecture')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--net_lr', type=float, default=0.1)
parser.add_argument('--param_lr', type=float, default=0.1)
parser.add_argument('--param_reg', type=float, default=0.01)
parser.add_argument('--tail', type=str, default='', help='tail message')
parser.add_argument('--save_dir', type=str, default='./saved_models/', help='save directory')
args = parser.parse_args()


## CUDA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('DEVICE:', device)

## Model Setup
model_dir = os.path.join(args.save_dir,
                            'sup_var',
                            f'{args.data}+{args.arch}',
                            f'samples{args.samples}'
                            f'epochs{args.epochs}'
                            f'_eps{args.eps}'
                            f'_netlr{args.net_lr}'
                            f'_paramlr{args.param_lr}'
                            f'_paramreg{args.param_reg}'
                            f'{args.tail}')
os.makedirs(model_dir, exist_ok=True)
utils.create_csv(model_dir, 'loss_mcr', ['loss_total', 'loss_discrimn', 'loss_compress', 'loss_reg'])
utils.create_csv(model_dir, 'acc_nearsub', ['epoch', 'acc_train', 'acc_test'])
utils.save_params(model_dir, vars(args))
print(model_dir)

## Experiment
trainset, testset, n_class = L.load_data(args.data)
X_train, y_train = F.get_samples(trainset, args.samples)
X_test, y_test = F.get_samples(testset, args.samples)
X_train = X_train.to(device)
X_test = X_test.to(device)
Pi = label_to_membership(y_train)


net = L.load_arch(args.data, args.arch)
net = nn.DataParallel(net).to(device)

criterion_mcr2var = MCR2Variational(args.eps, args.param_reg)
Us = nn.Parameter(
    torch.randn(n_class, 9, 9).float(),
    requires_grad=True
    ).to(device)
optimizer_net = optim.SGD(net.parameters(), lr=args.net_lr)
optimizer_Us = optim.SGD([Us], lr=args.param_lr)

## Training
for epoch in range(args.epochs):
    optimizer_net.zero_grad()
    optimizer_Us.zero_grad()

    # forward pass
    Z_train = net(X_train)
    # Z_train = torch.zeros(400, 128).float()
    # Z_train[:200, :64] = 1.
    # Z_train[200:, 64:] = 1.
    # Z_train = tF.normalize(Z_train)
    loss_obj, loss_R, loss_Rc, loss_reg_U = criterion_mcr2var(Z_train, Pi, Us)
    utils.append_csv(model_dir, 'loss_mcr', [-loss_obj.item(), loss_R, loss_Rc, loss_reg_U])
    print(epoch, loss_obj.item(), loss_R, loss_Rc, loss_reg_U)

    loss_obj.backward()
    optimizer_net.step()
    optimizer_Us.step()

    if epoch % 5 == 0:
        with torch.no_grad():
            Z_test = net(X_test)
        acc_train, acc_test = metrics_sup.nearsub(Z_train, y_train, Z_test, y_test, n_class, 1)
        plot.plot_heatmap(model_dir, Z_train.detach(), y_train.detach(), n_class, f'Ztrain{epoch}')
        # plot.plot_heatmap(model_dir, Z_test.detach(), y_test.detach(), n_class, f'Ztest{epoch}')
        plot.plot_loss_mcr2(model_dir, filename='loss_mcr')
        utils.save_array(model_dir, Us.detach().cpu(), f'Us_epoch{epoch}', 'Us')
        print(np.sum(Us.detach().cpu().numpy()[0] < 0), np.sum(Us.detach().cpu().numpy()[0] > 0))
        print(Us.detach().cpu().numpy()[0])

        for c in range(n_class):
            print(Us.detach().cpu().numpy()[c])
            plot.plot_array(model_dir, Us.detach().cpu().numpy()[c], f'U{c}')
