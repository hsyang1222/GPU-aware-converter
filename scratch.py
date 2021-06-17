#%% # ==================================================== #
# 
# ======================================================== #
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import InvertCnnConverter
# import copy

# class ConvBlock(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super().__init__()
#         layer = []
#         layer.append(nn.Conv2d(ch_in, ch_out, 3, 1, 1))
#         # layer.append(nn.BatchNorm2d(ch_out))
#         # layer.append(nn.ReLU())
#         layer.append(nn.Conv2d(ch_out, ch_out, 3, 1, 1))
#         # layer.append(nn.BatchNorm2d(ch_out))
#         # layer.append(nn.ReLU())
#         self.layer = nn.Sequential(*layer)
    
#     def forward(self, input):
#         return self.layer(input)

# class Down(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super().__init__()
#         self.pool = nn.MaxPool2d(2)
#         # self.conv = ConvBlock(ch_in, ch_out)
#         self.conv = nn.Conv2d(ch_in, ch_out, 3, 1, 1)
    
#     def forward(self, input):
#         out = self.pool(input)
#         out = self.conv(out)
#         return out

# class Up(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super().__init__()
#         self.unpool = nn.Upsample(scale_factor=2)
#         # self.conv = ConvBlock(ch_in, ch_out)
#         self.conv = nn.Conv2d(ch_in, ch_out, 3, 1, 1)
    
#     def forward(self, input, skip):
#         out = self.unpool(input)
#         out = torch.cat([out, skip], dim=1)
#         out = self.conv(out)
#         return out

# class Model(nn.Module):
#     def __init__(self, ch_in, ch_out):
#         super().__init__()
#         factor = 2
#         # self.conv_start = ConvBlock(ch_in, 64)
#         self.conv_start = nn.Conv2d(ch_in, 64, 3, 1, 1)
#         self.down1 = Down(64, 128 // factor)
#         self.up0 = Up(128, 64)
#         # self.conv_end = ConvBlock(64, ch_out)
#         self.conv_end = nn.Conv2d(64, ch_out, 3, 1, 1)
    
#     def forward(self, input):
#         out0 = self.conv_start(input)
#         out1 = self.down1(out0)
#         out = self.up0(out1, out0)
#         out = self.conv_end(out)
#         return out

# model = Model(1, 1)
# model2 = copy.deepcopy(model)
# model(torch.ones(2,1,64,64)).shape

# InvertCnnConverter.convert_module(model2, last_module_name="conv_end", inplace=True)
# model2(torch.ones(2,1,64,64)).shape

#%% # ==================================================== #
# debug
# ======================================================== #
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def debug_memory(title, v=False):
#     alloc = torch.cuda.memory_allocated(device)/1024/1024
#     max_alloc = torch.cuda.max_memory_allocated(device)/1024/1024
#     reserved = torch.cuda.memory_reserved(device)/1024/1024
#     max_reserved = torch.cuda.max_memory_reserved(device)/1024/1024
#     if v:
#         print(f'[{title:>10s}] alloc={alloc:.0f} / {max_alloc:.0f} MB, reserved={reserved:.0f} / {max_reserved:.0f} MB ')
#     return alloc, max_alloc

# device = torch.device("cuda:2")
# nn.Conv2d(1,1,3,1,1).to(device)(torch.ones(1,1,10,10).to(device)) # for cuda kernel init

# class MemNestedtic:
#     def __init__(self):
#         self.st = []
    
#     def tic(self, title):
#         a, ma = debug_memory(title)
#         self.st.append((title, ma, a))
#         level = len(self.st)
#         print(f'[{title:>10s}]{"  "*(level-1)} *')
        
#     def toc(self):
#         a, ma = debug_memory("")
#         level = len(self.st)
#         title, ma0, a0 = self.st.pop()
#         print(f'[{title:>10s}]{"  "*(level-1)} {a-a0:.0f}/{ma-ma0:.0f}')

# class Memtic:
#     def __init__(self):
#         self.st = []
    
#     def tic(self, title):
#         torch.cuda.empty_cache()
#         torch.cuda.reset_max_memory_allocated(device)
#         torch.cuda.reset_max_memory_cached(device)
#         a, ma = debug_memory(title)
#         self.st.append((title, ma, a))
        
#     def toc(self):
#         torch.cuda.empty_cache()
#         a, ma = debug_memory("")
#         level = len(self.st)
#         assert level <= 1
#         title, ma0, a0 = self.st.pop()
#         print(f'[{title:>10s}]{"  "*(level-1)} {a-a0:.0f}/{ma-ma0:.0f}')

# # mt = MemNestedtic()
# mt = Memtic()

# def f():
#     torch.cuda.empty_cache()
#     torch.cuda.reset_max_memory_allocated(device)
#     torch.cuda.reset_max_memory_cached(device)
#     # mt.tic("f+b")
#     def scope():
#         input = torch.ones(1,64,1024,1024, device=device, requires_grad=True)
#         model = nn.Conv2d(64,64,3,1,1).to(device)
#         mt.tic("forward") # 3.9
#         out = model(input) # 4.4
#         out = out.sum()
#         mt.toc() # 4.1
#         mt.tic("backward") # 
#         out.backward() # 5.0
#         mt.toc() # 4.4
#         print("-")
#     scope() # 4.4
#     # mt.toc()
#     _ = debug_memory("1", True)
#     torch.cuda.empty_cache()
#     _ = debug_memory("2", True)
#     torch.cuda.reset_max_memory_allocated(device)
#     torch.cuda.reset_max_memory_cached(device)
# # f()
# # _ = debug_memory("", True)




#%% # ==================================================== #
# 
# ======================================================== #
import torch
import bind

# torch.backends.cudnn.m.__dir__()
# torch.backends.cudnn.m._cudnn.__dir__()

# ---------------------------- 0 ---------------------------- #
# print("# ---------------------------- 0 ---------------------------- #")
# device = torch.device("cuda:0")
# out = torch.ones(1,1,10,10, device=device)
# input = torch.ones(1,1,10,10, device=device)
# print(bind.f(out, input))
# print("# ---------------------------- 0 end ---------------------------- #")

# ---------------------------- 1 ---------------------------- #
# print("# ---------------------------- 1 ---------------------------- #")
# device = torch.device("cuda:0")
# # device = torch.device("cpu")
# out = torch.ones(1,1,10,10, device=device)
# input = torch.ones(1,1,10,10, device=device)
# weight = torch.ones(1,1,3,3, device=device)
# bind.conv2d(out, input, weight)
# print("conv2d exit")
# print(out)
# print("end")

# import torch.nn.functional as F
# F.conv2d
# print("# ---------------------------- 1 end ---------------------------- #")

# ---------------------------- 2 ---------------------------- #
# print("# ---------------------------- 2 ---------------------------- #")
# device = torch.device("cuda:0")
# x = torch.ones(1,64,1024,1024, device=device)
# gy = torch.ones(1,64,1024,1024, device=device)
# gw = torch.zeros(64,64,3,3, device=device)
# print(x.device, x.shape)
# print(gy.device, gy.shape)
# print(gw.device, gw.shape)
# bind.conv2d_weight(gw, x, gy, 1, 1)
# print("backard exit")
# # print(gw)
# print("end")
# print("# ---------------------------- 2 end ---------------------------- #")

# ---------------------------- 3 ---------------------------- #
# print("# ---------------------------- 3 ---------------------------- #")
# device = torch.device("cuda:0")
# gw = torch.zeros(64,64,3,3, device=device)
# x = torch.ones(1,64,512,512, device=device)
# gy = torch.ones(1,64,510,510, device=device)
# print(x.device, x.shape)
# print(gy.device, gy.shape)
# print(gw.device, gw.shape)
# bind.conv2d_weight(gw, x, gy, 1, 0, device.index)
# print("backard exit")
# # print(gw)
# print("end")
# print("# ---------------------------- 3 end ---------------------------- #")


# ---------------------------- 4 ---------------------------- #
print("# ---------------------------- 4 ---------------------------- #")
from stitchable_conv.StitchableConv2d import StitchableConv2d

device = torch.device("cpu")
input= torch.ones(1,64,1024,1024, device=device, requires_grad=True)
model = StitchableConv2d(64, 64, 3, 1, 1, [512,512])
model = model.to(torch.device("cuda:0"))
out = model(input)
loss = out.sum()
loss.backward()
print("# ---------------------------- 4 end ---------------------------- #")

# import torch
# x = torch.ones(3).cuda(0)
# x.device.index
