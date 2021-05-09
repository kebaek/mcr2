def load_data(option, data_dir='./data/', *args, **kwargs):
    if option == 'gaussians':
        from datasets.gaussians import Gaussian3D
        trainset = Gaussian3D(3, perclass=1000, noise=.1)
        testset = Gaussian3D(3, perclass=1000, noise=.1)
        return trainset, testset, 3
    if option == 'mnist':
        from datasets.mnist import mnist
        return mnist(num_classes=10, data_dir=data_dir)
    if option == 'mnist_5class':
        from datasets.mnist import mnist
        return mnist(num_classes=5, data_dir=data_dir)
    if option == 'mnist_2class':
        from datasets.mnist import mnist
        return mnist(num_classes=2, data_dir=data_dir)
    raise ValueError(f'Data option not found: {option}')

def load_arch(data, option, *args, **kwargs):
    if data == 'gaussians':
        if option == 'mlp':
            from architectures.gaussians.mlp import MLP
            return MLP(3)
        if option == 'simple_cnn':
            from architectures.simple_cnn import SimpleCNN
            return SimpleCNN(3) #TODO
    if data in ['mnist', 'mnist_5class', 'mnist_2class']:
        if option == 'simple':
            from architectures.mnist.simple import Net
            return Net()
        if option == 'resnet10':
            from architectures.mnist.resnet_mnist import ResNet10MNIST
            return ResNet10MNIST()
    raise ValueError(f'Architecture option not found: {option}')