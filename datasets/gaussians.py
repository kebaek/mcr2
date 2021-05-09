import torch
import torch.nn.functional as F


def load_gaussians(nclass, perclass=50, noise=.1):
    # nclass is number of gaussians to load
    assert nclass <= 6
    d = 3
    means = (torch.tensor([1,0,0]), torch.tensor([0,1,0]),
             torch.tensor([0,0,1]), torch.tensor([-1,0,0]),
             torch.tensor([0,-1,0]), torch.tensor([0,0,-1]))
    full_samples = torch.zeros(nclass*perclass, d)
    label = torch.zeros(nclass*perclass)
    for i in range(nclass):
        mean = means[i]
        samples = mean.unsqueeze(0) + noise*torch.randn(perclass, d)
        class_inds = list(range(i*perclass, (i+1)*perclass))
        full_samples[class_inds] = samples
        label[class_inds] = i
    full_samples = F.normalize(full_samples)
    return full_samples, label, nclass

class Gaussian3D(torch.utils.data.Dataset):
    def __init__(self, nclass, perclass=50, noise=.1):
        super(Gaussian3D, self).__init__()
        self.data, self.targets, self.nclass = load_gaussians(nclass, perclass, noise)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]