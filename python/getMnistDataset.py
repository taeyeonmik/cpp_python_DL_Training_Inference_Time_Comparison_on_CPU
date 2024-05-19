import torch
from torchvision import datasets, transforms

# Download MNIST dataset
train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transforms.ToTensor())
# Save dataset
torch.save(train_dataset, '../train_dataset.pt')
torch.save(test_dataset, '../test_dataset.pt')