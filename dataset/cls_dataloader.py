import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms.transforms import Resize

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Normalize training set together with augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
])

train_dataloader = DataLoader(
    datasets.cifar.CIFAR100(root=r' ', train=True, transform=transform_train, download=True),
    batch_size=128,
    shuffle=True,
    num_workers=4
)

test_dataloader = DataLoader(
    datasets.cifar.CIFAR100(root=r'', train=False, transform=transform_test, download=True),
    batch_size=128,
    shuffle=False,
    num_workers=4
)