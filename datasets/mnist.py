import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .utils_data import filter_class



def mnist(num_classes, data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    if num_classes == 2:
        trainset, num_classes = filter_class(trainset, [0, 1])
        testset, _ = filter_class(testset, [0, 1])
    elif num_classes == 5:
        trainset, num_classes = filter_class(trainset, [0, 1, 2, 3, 4])
        testset, _ = filter_class(testset, [0, 1, 2, 3, 4])
    elif num_classes == 10:
        pass
    else:
        raise ValueError('number of classes undefined.')
    return trainset, testset, num_classes
