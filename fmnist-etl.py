#!/usr/bin/env python

import torch
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST", # Where data is/will be stored
    train=True,			# flag to say that this is a training set
    download=True,		# Download if not present
    transform=transforms.Compose([transforms.ToTensor()]) # Transoform operation
    )

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)


