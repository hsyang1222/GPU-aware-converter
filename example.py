from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule, UnetModule
from fastmri.data import transforms, mri_data
import pathlib
import torch
import matplotlib.pyplot as plt
import numpy as np


CHALLENGE = 'singlecoil'
MASK_TYPE = 'random'
center_fractions = [0.08]
accelerations = [4]


mask = create_mask_for_mask_type(
        MASK_TYPE, center_fractions, accelerations
    )

train_transform = UnetDataTransform(CHALLENGE, mask_func=mask, use_seed=False)

dataset = mri_data.SliceDataset(
    root=pathlib.Path(
      './fastmri_data/singlecoil_val'
    ),
    transform=train_transform,
    challenge='singlecoil'
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
image, target, _, _, _, _, _ = next(iter(dataloader))



import InvertCnnConverter
import torch
import Unet

import importlib
importlib.reload(InvertCnnConverter)

plain_model = Unet.UNet(n_channels=1, n_classes=1)

#InvertCnnConverter.top_forward_to_checkpoint(plain_model, last_module_name='outc')
InvertCnnConverter.convert_module(plain_model, last_module_name='outc', inplace=True)
invert_model = plain_model

device = 'cuda:0'

invert_model = invert_model.to(device)
data = image.view(-1,1,320,320).to(device)
target = target.to(device)

print(invert_model)



criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(invert_model.parameters(), lr=1e-3)
with torch.autograd.set_detect_anomaly(True) : 
    for epoch in range(1000) : 
        result = invert_model(data)
        loss = criterion(result, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(epoch, loss.item())