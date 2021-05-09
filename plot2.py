import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import  matplotlib.cm as cm

import load as L
import functional as F
import utils



def plot_loss_mcr2(model_dir, filename='loss_mcr'):
    """Plot theoretical loss and empirical loss. """
    ## extract loss from csv
    file_dir = os.path.join(model_dir, 'csv', f'{filename}.csv')
    data = pd.read_csv(file_dir)
    loss_total = data['loss_total'].ravel()
    loss_discrimn = data['loss_discrimn'].ravel()
    loss_compress = data['loss_compress'].ravel()
    loss_reg = data['loss_reg'].ravel()

    ## Theoretical Loss
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    num_iter = np.arange(len(loss_total))
    ax.plot(num_iter, loss_discrimn - loss_compress, label=r'$\Delta R$', 
                color='green', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, loss_discrimn, label=r'$R$', 
                color='royalblue', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, loss_compress, label=r'$R^c$', 
                color='coral', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, loss_discrimn - loss_compress - loss_reg, label=r'objective', 
                color='black', linewidth=1.0, alpha=0.8) 
    ax.plot(num_iter, loss_reg, label=r'$0.5*\gamma||\cdot||^2$', 
                color='purple', linewidth=1.0, alpha=0.8)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Number of iterations')
    ax.legend()
    # ax.legend(loc='lower right', prop={"size": 15}, ncol=3, framealpha=0.5)
    ax.set_title("MCR2 Loss")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    ## create saving directory
    loss_dir = os.path.join(model_dir, 'figures', 'loss')
    os.makedirs(loss_dir, exist_ok=True)
    file_name = os.path.join(loss_dir, 'loss_mcr.png')
    plt.savefig(file_name, dpi=400)
    plt.close()
    print("Plot saved to: {}".format(file_name))


def plot_membership(model_dir, pi, name, title=''):
    save_dir = os.path.join(model_dir, 'figures', 'memberships')
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(pi, cmap='Blues')
    ax.set_title(title)
    fig.colorbar(im, pad=0.02, drawedges=0)
    fig.savefig(os.path.join(save_dir, f'{name}.png'))
    plt.close()

def plot_array(model_dir, array, name, folder=''):
    save_dir = os.path.join(model_dir, 'figures', 'array', folder)
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(array, cmap='Blues')
    ax.set_title(name)
    fig.colorbar(im, pad=0.02, drawedges=0)
    fig.savefig(os.path.join(save_dir, f'{name}.png'))
    plt.close()
    print('Plot saved to:', os.path.join(save_dir, f'{name}.png'))

def plot_3d(model_dir, Z, y, name):
    colors = np.array(['green', 'blue', 'red'])
    savedir = os.path.join(model_dir, 'figures', '3d')
    os.makedirs(savedir, exist_ok=True)
    colors = np.array(['forestgreen', 'royalblue', 'brown'])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=colors[y], cmap=plt.cm.Spectral, s=200.0)
    # Z, _ = F.get_n_each(Z, y, 1)
    # for c in np.unique(y):
        # ax.quiver(0.0, 0.0, 0.0, Z[c, 0], Z[c, 1], Z[c, 2], length=1.0, normalize=True, arrow_length_ratio=0.05, color='black')
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.5)
    ax.xaxis._axinfo["grid"]['color'] =  (0,0,0,0.1)
    ax.yaxis._axinfo["grid"]['color'] =  (0,0,0,0.1)
    ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,0.1)
    [tick.label.set_fontsize(24) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(24) for tick in ax.yaxis.get_major_ticks()]
    [tick.label.set_fontsize(24) for tick in ax.zaxis.get_major_ticks()]
    ax.view_init(20, 15)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"scatter3d-{name}.jpg"), dpi=200)
    plt.close()

def plot_heatmap(model_dir, features, labels, num_classes, filename):
    """Plot heatmap of cosine simliarity for all features. """
    features_sort, _ = utils.sort_dataset(features, labels, num_classes, stack=False)
    features_sort_ = np.vstack(features_sort)
    sim_mat = np.abs(features_sort_ @ features_sort_.T)

    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True)
    im = ax.imshow(sim_mat, cmap='Blues')
    fig.colorbar(im, pad=0.02, drawedges=0)
    # ax.set_xticks(np.linspace(0, 50000, 6))
    # ax.set_yticks(np.linspace(0, 50000, 6))
    [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()
    save_dir = os.path.join(model_dir, 'figures', 'heatmaps')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f"heatmat_{filename}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_acc(model_dir, filename):
    metrics = ['acc', 'nmi', 'fd', 'nnz']
    metrics_display = ['SC Acc', 'SC NMI', 'SPE', 'NNZ']
    n_metrics = len(metrics)

    df = pd.read_csv(os.path.join(model_dir, 'csv', f'{filename}.csv'))

    fig, ax = plt.subplots(nrows=n_metrics, figsize=(7, 10), sharey=False, sharex=True, dpi=400)
    for row, metric in enumerate(metrics):
        acc_arr = df[metric].ravel()
        step = np.arange(acc_arr.size)
        ax[row].plot(step, acc_arr)
        ax[row].set_ylabel(metrics_display[row])
        ax[row].set_xlabel('Epoch')
    fig.tight_layout()
    save_dir = os.path.join(model_dir, 'figures', 'acc')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f"{filename}.png")
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_pca(model_dir, features, labels, num_classes):
    """Plot PCA of learned features. """
    ## create save folder
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    features_sort, _ = utils.sort_dataset(features, labels, num_classes)
    colors = cm.rainbow(np.linspace(0, 1, num_classes))
    pca = PCA(n_components=None, svd_solver='full').fit(features)
    ax.plot(np.arange(len(pca.singular_values_)), pca.singular_values_, 
                color='black', marker='x', alpha=0.4, label='all')
    for class_i in range(num_classes):
        pca = PCA(n_components=None, svd_solver='full').fit(features_sort[class_i])
        ax.plot(np.arange(len(pca.singular_values_)), pca.singular_values_, 
                    color=colors[class_i], marker='o', alpha=0.5, label=f'{class_i}')
    ax.legend()
    ax.set_xlabel('Components')
    ax.set_ylabel('Singular Values')
    fig.tight_layout()

    pca_dir = os.path.join(model_dir, 'figures', 'pca')
    os.makedirs(pca_dir, exist_ok=True)
    filename = os.path.join(pca_dir, 'pca.png')
    fig.savefig(filename)
    plt.close()
    print("Plot saved to: {}".format(filename))
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_dir', type=str, required=True, help='base directory for saving PyTorch model.')
    parser.add_argument('--epoch', type=int, required=True, help='which epoch for evaluation')
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--pi', action='store_true')
    parser.add_argument('--pi_sc', action='store_true')
    parser.add_argument('--data_dir', default='./data/', help='path to dataset')
    args = parser.parse_args()

    params = utils.load_params(args.model_dir)
    trainset, testset, num_classes = L.load_data(params['data'])

    X_train = np.load(os.path.join(args.model_dir, 'features', 'X_train.npy'))
    y_train = np.load(os.path.join(args.model_dir, 'features', 'y_train.npy'))
    Z_train = np.load(os.path.join(args.model_dir, 'features', 'Z_train.npy'))
    Pi = np.load(os.path.join(args.model_dir, 'membership', f'epoch{args.epoch}.npy'))

    if args.pca:
        plot_pca(args.model_dir, Z_train, y_train, num_classes)
    if args.pi:
        plot_membership(args.model_dir, Pi, f'epoch{args.epoch}')
    if args.pi_sc:
        from metrics_cluster import spectral_clustering_metrics
        acc_lst, _, _, _, pred_lst = spectral_clustering_metrics(Pi, num_classes, y_train)
        Pi_srt = Pi[:, np.argsort(pred_lst[-1])]
        counts = np.unique(pred_lst[-1], return_counts=True)[1].tolist()
        plot_membership(args.model_dir, Pi_srt, f'epoch{args.epoch}_srt', title=f'{counts}')
