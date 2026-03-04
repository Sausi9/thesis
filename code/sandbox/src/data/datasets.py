from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data():
    # 1. Define Transforms (Critical for performance)
    # MNIST images are 28x28. ToTensor() scales pixels from [0, 255] to [0.0, 1.0].
    # Normalize uses the Mean (0.1307) and Std Dev (0.3081) of the MNIST dataset.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    DATA_ROOT = "/Users/audun/Desktop/MSc/thesis/code/sandbox/data"

    # 2. Download/Load the Datasets
    mnist_train = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    # 3. Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False)

    return train_loader, test_loader
