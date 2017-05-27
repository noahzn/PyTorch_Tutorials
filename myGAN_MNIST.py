import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_set = torchvision.datasets.MNIST(root='./data/',
                                       train=True,
                                       transform=transform,
                                       download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=40,
                                           shuffle=True,
                                           num_workers=2)



