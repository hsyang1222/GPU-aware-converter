#%% # ==================================================== #
# 실험 결과: https://docs.google.com/spreadsheets/d/16Bkg4N49t9f-iMfTX0zLULOr3--kQ4TBsn7z8epB7G8/edit#gid=0
# ======================================================== #
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import InvertCnnConverter
from tqdm import tqdm
from stitchable_conv.StitchableConv2d import StitchableConv2d
device = torch.device("cuda:2")
nn.Conv2d(1,1,3,1,1).to(device)(torch.ones(1,1,10,10).to(device)) # for cuda kernel init

def debug_memory(title, v=False):
    alloc = torch.cuda.memory_allocated(device)/1024/1024
    max_alloc = torch.cuda.max_memory_allocated(device)/1024/1024
    reserved = torch.cuda.memory_reserved(device)/1024/1024
    max_reserved = torch.cuda.max_memory_reserved(device)/1024/1024
    if v:
        print(f'[{title:>10s}] alloc={alloc:.0f} / {max_alloc:.0f} MB, reserved={reserved:.0f} / {max_reserved:.0f} MB ')
    return alloc, max_alloc

class Memlog:
    def __init__(self, name):
        self.name = name

    def __call__(self, fn):
        def wrap(*args, **kwargs):
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated(device)
            torch.cuda.reset_max_memory_cached(device)    
            a0, ma0 = debug_memory("")
            fn(*args, **kwargs)
            a1, ma1 = debug_memory("")
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated(device)
            torch.cuda.reset_max_memory_cached(device)
            # print(f'[{self.name:>10s}]: {ma1-ma0:.0f}')
            print(f'{ma1-ma0:.0f}')
            return ma1-ma0
        return wrap

class Model(nn.Module):
    def __init__(self, ch, n):
        super().__init__()
        self.n = n
        for i in range(n):
            self.__setattr__(f'conv{i}', nn.Conv2d(ch, ch, 3, 1, 1))

    def forward(self, x):
        for i in range(self.n):
            x = self.__getattr__(f'conv{i}')(x)
        return x
        
# class RestoreStitchableConv(nn.Module):
#     def __init__(self, ch_in, ch_out, k, s, p, fs):
#         super().__init__()
#         self.conv = StitchableConv2d(ch_in, ch_out, k, s, p, fs)

#     def forward(self, x):
#         assert x.device == torch.device("cpu")
#         x = self.conv(x)
#         return x

class StitchableModel(nn.Module): # fetch size는 512,512를 사용중
    def __init__(self, ch, n):
        super().__init__()
        self.n = n
        for i in range(n):
            self.__setattr__(f'conv{i}', StitchableConv2d(ch, ch, 3, 1, 1, [512,512]))

    def forward(self, x):
        for i in range(self.n):
            x = self.__getattr__(f'conv{i}')(x)
        return x

@Memlog("conv")
def f(ch, size, num_layer):
    input = torch.ones(1,ch,size,size, device=device, requires_grad=True)
    # model = nn.Conv2d(ch,ch,3,1,1).to(device)
    model = Model(ch, num_layer).to(device)
    out = model(input)
    loss = out.sum()
    loss.backward()

@Memlog("inv")
def g(ch, size, num_layer):
    input = torch.ones(1,ch,size,size, device=device, requires_grad=True)
    # model = nn.Conv2d(ch,ch,3,1,1).to(device)
    model = Model(ch, num_layer).to(device)
    InvertCnnConverter.conv2d_to_invertible(model)
    model = model.to(device)
    out = model(input)
    loss = out.sum()
    loss.backward()

@Memlog("stitch")
def h(ch, size, num_layer):
    input = torch.ones(1,ch,size,size, requires_grad=True)
    # model = nn.Conv2d(ch,ch,3,1,1).to(device)
    # model = Model(ch, num_layer).to(device)
    model = StitchableModel(ch, num_layer).to(device)
    out = model(input)
    loss = out.sum()
    loss.backward()

print(f'# ---------------------------- 1 --------------------------- #')
# f(64, 1024, 1)
# f(64, 512, 1)
# f(32, 512, 1)
# for i in range(1, 5):
#     f(64, 1024, i)
# f(64, 1024, 10)

print(f'# ---------------------------- 2 ---------------------------- #')
# g(64, 1024, 1)
# g(64, 512, 1)
# g(32, 512, 1)
# for i in range(1, 5):
#     g(64, 1024, i)
# g(64, 1024, 10)

print(f'# ---------------------------- 3 ---------------------------- #')
# h(64, 1024, 1)
# h(64, 512, 1)
# h(32, 512, 1)
# for i in range(1, 5):
#     h(64, 1024, i)
for i in tqdm(range(2)):
    h(64, 1024, 10)

print(f'# ---------------------------- 4 ---------------------------- #')
# for i in range(1, 5):
#     f(64, 512, i)

# for i in range(1, 5):
#     g(64, 512, i)

# for i in range(1, 5):
#     h(64, 1024, i)

print(f'# ---------------------------- 5 ---------------------------- #')
# f(64, 2048, 1)
# f(64, 1024, 1)
# f(64, 512, 1)
# f(64, 256, 1)