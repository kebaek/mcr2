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
    """Equation 10 in writeup. """
    def __init__(self, eps, mu):
        super(MCR2Variational, self).__init__()
        self.eps = eps
        self.mu = mu

    def loss_discrimn(self, Z):
        d, n = Z.shape
        I = torch.eye(d).to(Z.device)
        return 0.5 * torch.logdet(I + d / (n * self.eps) * Z @ Z.T)

    def loss_compress(self, Z, Pi, A):
        d, m = Z.shape
        _, k= Pi.shape
        I = torch.eye(d).to(Z.device)
        compress_loss = 0.
        r = torch.nn.ReLU()
        ones = torch.ones(A.shape[1])
        for j in range(k):
            trPi = torch.sum(Pi[:,j])
            scalar = d / (trPi * eps)
            log_det = torch.sum(torch.log(ones + scalar * r(A[j])))
            compress_loss += log_det * trPi / m
        return compress_loss / 2

    def reg_UA(self, Z, Pi, A, U):
        _, k = Pi.shape
        matrix_loss = 0.
        U = torch.nn.functional.normalize(U, dim = 0)
        r = torch.nn.ReLU()
        for j in range(k):
            matrix_loss += torch.norm(Z@Pi[:,j].diag()@Z.T - U@torch.diag(r(A[j]))@U.T)**2
        return  matrix_loss / 2

    def forward(self, Z, Pi, A, U):
        loss_R = self.loss_discrimn(Z.T)
        loss_Rc = self.loss_compress(Z.T, Pi, A)
        loss_reg_U = self.mu * self.reg_UA(Z.T, Pi, A, U)
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
parser.add_argument('--net_lr', type=float, default=0.001)
parser.add_argument('--param_lr', type=float, default=0.001)
parser.add_argument('--tail', type=str, default='', help='tail message')
parser.add_argument('--save_dir', type=str, default='./saved_models/', help='save directory')
parser.add_argument('--tol', type=float, default=0)
parser.add_argument('--mu', type=float, default=1.0)

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
                            f'_paramreg{args.mu}'
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

mcr = MCR2Variational(args.eps, args.mu)
true_mcr = MaximalCodingRateReduction(args.eps)

net = L.load_arch(args.data, args.arch)
net = nn.DataParallel(net).to(device)
init_U = []
init_A = []
with torch.no_grad():
    Z_train = net(X_train)
    total_r = 0
    for j in range(n_class):
        U, S, _ = torch.linalg.svd(Z_train.T @ Pi[:, j].diag() @ Z_train)
        r = 1
        error = float('inf')
        while True:
            error = torch.norm(Z@Pi[:,j].diag()@Z.T - U[:,:r]@torch.diag(r(S[:r]))@U[:,:r].T)**2
            if error <= args.tol:
                break
            r += 1
        total_r += r
        init_U.append(U[:,:r])
        init_A.append(S[:r])
print('r: %d'%total_r)
U = torch.cat(init_Us, dim=1)
A = torch.zeros((n_class, total_r))
r = 0
for i in range(n_class):
    A[i] = torch.tensor([0]*total_r + init_A[i].tolist() + [0]*(total_r - init_A[i].shape[0]))
    r += init_A[i].shape[0]

U = nn.Parameter(
    U,
    requires_grad=True
    ).to(device)
A = nn.Parameter(
    A,
    requires_grad=True
    ).to(device)
optimizer_net = optim.SGD(net.parameters(), lr=args.net_lr)
# optimizer_net = optim.Adadelta(net.parameters(), lr=args.net_lr)
optimizer_UA = optim.SGD([U, A], lr=args.param_lr)

## Training
for epoch in range(args.epochs):
    optimizer_net.zero_grad()

    # forward pass
    Z_train = net(X_train)
    loss_obj, loss_R, loss_Rc, loss_reg_U = mcr(Z_train, Pi, A, U)
    true_loss, loss_empi, loss_theo = criterion(Z_train, Pi, num_classes=trainset.num_classes)
    utils.append_csv(model_dir, 'loss_var', [-loss_obj.item(), loss_R, loss_Rc, loss_reg_U])
    print(epoch, -loss_obj.item(), loss_R, loss_Rc, loss_reg_U)
    utils.append_csv(model_dir, 'loss_true', [-loss_obj.item(), loss_R, loss_Rc, loss_reg_U])
    print(epoch, -loss_obj.item(), loss_empi[0], loss_empi[1])


    loss_obj.backward()
    optimizer_net.step()

    for step in range(20):
        optimizer_UA.zero_grad()
        loss_obj, loss_R, loss_Rc, loss_reg_U = criterion_mcr2var(Z_n, Pi, A, U)
        loss_obj.backward()
        optimizer_UA.step()

    if epoch % 200 == 0:
        with torch.no_grad():
            Z_test = net(X_test)
        #acc_train, acc_test = metrics_sup.nearsub(Z_train, y_train, Z_test, y_test, n_class, 1)
        #plot.plot_heatmap(model_dir, Z_train.detach(), y_train.detach(), n_class, f'Ztrain{epoch}')
        # plot.plot_heatmap(model_dir, Z_test.detach(), y_test.detach(), n_class, f'Ztest{epoch}')
        # plot.plot_loss_mcr2(model_dir, filename='loss_mcr')
        # plot.plot_loss_mcr2_2(model_dir, filename='loss_mcr')
        # utils.save_array(model_dir, U.detach().cpu(), f'U_epoch{epoch}', 'U')
        # utils.save_array(model_dir, A.detach().cpu(), f'A_epoch{epoch}', 'A')

        # for c in range(n_class):
        #     plot.plot_array(model_dir, Us.detach().cpu().numpy()[c], f'U{c}')
utils.save_ckpt(model_dir, net, epoch)
