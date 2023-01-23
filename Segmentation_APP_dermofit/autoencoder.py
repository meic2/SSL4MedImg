import torch
from torch import nn
import torch.nn.functional as F

class AutoencoderReLu(nn.Module):
    def __init__(self):
        super(AutoencoderReLu, self).__init__()
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

class AutoencoderGeLu(nn.Module):
    def __init__(self):
        super(AutoencoderGeLu, self).__init__()
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
        x = F.gelu(self.enc1(x))
        x = F.gelu(self.encmid(x))
        x = F.gelu(self.enc2(x))
        x = F.gelu(self.enc3(x))
        x = F.gelu(self.enc4(x))
        x = F.gelu(self.enc5(x))

        x = F.gelu(self.dec1(x))
        x = F.gelu(self.dec2(x))
        x = F.gelu(self.dec3(x))
        x = F.gelu(self.dec4(x))
        x = F.gelu(self.dec5(x))
        return x
