'''
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule, UnetModule
from fastmri.data import transforms, mri_data



def load_data_target () : 
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

    return image, target

'''

import InvertCnnConverter
import torch
import Unet

def load_sample() : 
    image = torch.load("./demo/fastMRIsample_image.tensor")
    target = torch.load("./demo/fastMRIsample_target.tensor")
    return image, target


#image, target = load_data_target()
image, target = load_sample()

plain_model = Unet.UNet(n_channels=1, n_classes=1)

device = 'cuda:0'

plain_model = plain_model.to(device)
data = image.view(-1,1,320,320).to(device)
target = target.to(device)

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(plain_model.parameters(), lr=1e-2)
with torch.autograd.set_detect_anomaly(True) : 
    for epoch in range(1000) : 
        result = plain_model(data)
        loss = criterion(result, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        print(epoch, loss.item())