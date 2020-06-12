import torch
from torch.utils.data import Dataset, Dataloader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_cifar10(batch=128):
    train_loader = DataLoader(
        datasets.CIFAR10('./',
                        train=True,
                        dowanload=True,
                        transform=transforms.Compose([
                                                    transforms.RandomHorizontialFlip(p=0.5),
                                                    transforms.RandomAffine(degree=0.2, scale=(0.8, 1.2)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        mean=[0.485, 0.456, 0.406],
                                                        std=[0.229,0.224, 0.225]
                                                    )
                        ])),
                        batch_size=batch,
                        shuffle=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10('./',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]
                                                    )
                        ])),
                        batch_size=batch,
                        shuffle=True
    )

    return {'train': train_loader, 'test': test_loader}
