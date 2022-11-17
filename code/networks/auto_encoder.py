import pandas as pd
import os
from tqdm import tqdm
import random
import itertools
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn, optim
import time
import copy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import segmentation_models_pytorch as smp

import timm
import math

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=224, out_features=15360) 
        self.encmid = nn.Linear(in_features=15360, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=16)

        # decoder 
        self.dec1 = nn.Linear(in_features=16, out_features=32)
        self.dec2 = nn.Linear(in_features=32, out_features=64)
        self.dec3 = nn.Linear(in_features=64, out_features=128)
        self.dec4 = nn.Linear(in_features=128, out_features=256)
        self.dec5 = nn.Linear(in_features=256, out_features=224) 

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.encmid(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        return x

 
 
if __name__ == "__main__":
    input = torch.rand(8, 3, 480, 480)
    model = Autoencoder()
    out = model(input)
    print(f"out.shape: ", out.shape) # out.shape:  torch.Size([8, 3, 480, 480])
    criterion = nn.MSELoss()
    print(f"criterion(out, input): {criterion(out, input)}") # criterion(out, input): 0.31674283742904663