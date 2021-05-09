import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC

import load as L
import functional as F
import utils



def svm(train_features, train_labels, test_features, test_labels):
    svm = LinearSVC(verbose=0, random_state=10)
    svm.fit(train_features, train_labels)
    acc_train = svm.score(train_features, train_labels)
    acc_test = svm.score(test_features, test_labels)
    print("SVM: {}".format(acc_test))
    return acc_train, acc_test


def knn(train_features, train_labels, test_features, test_labels, k=5):
    sim_mat = train_features @ train_features.T
    topk = sim_mat.topk(k=k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc_train = compute_accuracy(test_pred, train_labels)

    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(k=k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc_test = compute_accuracy(test_pred, test_labels)
    print("kNN: {}, {}".format(acc_train, acc_test))
    return acc_train, acc_test


def nearsub(train_features, train_labels, test_features, test_labels, 
            num_classes, n_comp=10, return_pred=False):
    train_scores, test_scores = [], []
    classes = np.arange(num_classes)
    features_sort, _ = utils.sort_dataset(train_features, train_labels, 
                                          classes=classes, stack=False)           
    fd = features_sort[0].shape[1]
    for j in classes:
        _, _, V = torch.svd(features_sort[j])
        components = V[:, :n_comp].T
        subspace_j = torch.eye(fd) - components.T @ components
        train_j = subspace_j @ train_features.T
        test_j = subspace_j @ test_features.T
        train_scores_j = torch.linalg.norm(train_j, ord=2, axis=0)
        test_scores_j = torch.linalg.norm(test_j, ord=2, axis=0)
        train_scores.append(train_scores_j)
        test_scores.append(test_scores_j)
    train_pred = torch.stack(train_scores).argmin(0)
    test_pred = torch.stack(test_scores).argmin(0)
    if return_pred:
        return train_pred.numpy(), test_pred.numpy()
    train_acc = compute_accuracy(classes[train_pred], train_labels.cpu().numpy())
    test_acc = compute_accuracy(classes[test_pred], test_labels.cpu().numpy())
    print('SVD: {}, {}'.format(train_acc, test_acc))
    return train_acc, test_acc

def argmax(train_features, train_labels, test_features, test_labels):
    train_pred = train_features.argmax(1)
    train_acc = compute_accuracy(train_pred, train_labels)
    test_pred = test_features.argmax(1)
    test_acc = compute_accuracy(test_pred, test_labels)
    return train_acc, test_acc

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    if type(y_pred) == torch.Tensor:
        n_wrong = torch.count_nonzero(y_pred - y_true).item()
    elif type(y_pred) == np.ndarray:
        n_wrong = np.count_nonzero(y_pred - y_true)
    else:
        raise TypeError("Not Tensor nor Array type.")
    n_samples = len(y_pred)
    return 1 - n_wrong / n_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_dir', type=str, required=True, help='base directory for saving PyTorch model.')
    parser.add_argument('--epoch', type=int, required=True, help='which epoch for evaluation')
    parser.add_argument('--svm', help='evaluate using SVM', action='store_true')
    parser.add_argument('--knn', help='evaluate using kNN measuring cosine similarity', action='store_true')
    parser.add_argument('--nearsub', help='evaluate using Nearest Subspace', action='store_true')
    

    parser.add_argument('--k', type=int, default=5, help='top k components for kNN (default:5)')
    parser.add_argument('--n_comp', type=int, default=10, help='number of components (default: 10)')
    parser.add_argument('--save', action='store_true', help='save labels')
    parser.add_argument('--data_dir', default='./data/', help='path to dataset')
    args = parser.parse_args()

    ## CUDA
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)

    ## load model
    params = utils.load_params(args.model_dir)
    net = L.load_arch(params['data'], params['arch'])
    net = nn.DataParallel(net)
    net = utils.load_ckpt(args.model_dir, f'model-epoch{args.epoch}', net)
    net = net.to(device)
    net.eval()
    
    trainset, testset, num_classes = L.load_data(params['data'])
    trainloader = DataLoader(trainset, batch_size=100)
    testloader = DataLoader(testset, batch_size=100)
    train_features, train_labels = F.get_features(net, trainloader, device=device)
    test_features, test_labels = F.get_features(net, testloader, device=device)

    if args.svm:
        svm(train_features.numpy(), train_labels.numpy(), test_features.numpy(), test_labels.numpy())
    if args.knn:
        knn(train_features, train_labels, test_features, test_labels, args.k)
    if args.nearsub:
        nearsub(train_features, train_labels, test_features, test_labels, num_classes, args.n_comp)
