import torch
import torch.nn as nn
import memcnn
from collections import OrderedDict
import copy

class UpInvertibleBlock(nn.Module):
     def __init__(self, in_c, out_c, conv_param):
        super().__init__()
        scale = self.make_scale(in_c, out_c)
        self.out_channels = out_c
        self.upscale = torch.nn.Upsample(scale_factor=scale, mode='bilinear')
        #self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        fm_input_size = out_c // 2
        gm_input_size = out_c - fm_input_size
        k,s,p,d,g = conv_param
        self.conv2 = memcnn.InvertibleModuleWrapper(fn= \
                             memcnn.AdditiveCoupling(
                                    Fm=torch.nn.Conv2d(fm_input_size, fm_input_size, k, s, p, d, g),
                                    Gm=torch.nn.Conv2d(gm_input_size, gm_input_size, k, s, p, d, g),
                             ), keep_input=False, keep_input_inverse=False)
    
     def forward(self, x) : 
        x_shape = x.shape
        x = self.upscale(x)
        x = x.view(x_shape[0],self.out_channels, x_shape[2], x_shape[3])
        x = self.conv2(x)
        return x
    
     def make_scale(self, in_c, out_c) : 
        multiple = out_c // in_c
        msq = multiple ** 0.5
        if msq == int(msq) : 
            scale = (msq, msq)
        else : 
            scale = (multiple, 1)
        return scale

class DownInvertibleBlock(nn.Module):
    def make_scale(self, in_c, out_c) : 
        multiple = out_c // in_c
        return multiple
    
            
    def forward(self, x) : 
        x = self.conv2(x)
        x = self.down_channel(x)
        return x
    
    def __init__(self, in_c, out_c, conv_param):
        super().__init__()
        fm_input_size = in_c // 2
        gm_input_size = in_c - fm_input_size
        k,s,p,d,g = conv_param
        self.conv2 = memcnn.InvertibleModuleWrapper(fn= \
                             memcnn.AdditiveCoupling(
                                    Fm=torch.nn.Conv2d(fm_input_size, fm_input_size, k, s, p, d, g),
                                    Gm=torch.nn.Conv2d(gm_input_size, gm_input_size, k, s, p, d, g),
                             ), keep_input=False, keep_input_inverse=False)
    
        
        scale = self.make_scale(in_c, out_c)
        kernel_size = (scale, 1, 1)
        self.down_channel = torch.nn.MaxPool3d(kernel_size)
    
    
    

def conv2d_to_invertible(block, inplace=True) : 
    replace_modules = copy.deepcopy(block._modules)     
    #print("before for loop", replace_modules)
    for i, (name, module) in enumerate(block.named_modules()) : 
        #print(i, name)
        if '.' not in name and isinstance(module, torch.nn.Conv2d) \
             and not isinstance(module, memcnn.InvertibleModuleWrapper):
            in_c = module.in_channels
            out_c = module.out_channels
            k = module.kernel_size
            s = module.stride
            p = module.padding
            d = module.dilation
            t = module.transposed
            op = module.output_padding
            g = module.groups
            if in_c == out_c : 
                #print(name, module, "\t\t-->")
                fm_input_size = in_c // 2
                gm_input_size = in_c - fm_input_size
                conv2d = memcnn.InvertibleModuleWrapper(fn= \
                             memcnn.AdditiveCoupling(
                                    Fm=torch.nn.Conv2d(fm_input_size, fm_input_size, k, s, p, d, g),
                                    Gm=torch.nn.Conv2d(gm_input_size, gm_input_size, k, s, p, d, g),
                             ), keep_input=False, keep_input_inverse=False)
                replace_modules[name] = conv2d
                #print(conv2d)
            else : 
                if in_c < out_c : 
                    ub = UpInvertibleBlock(in_c, out_c, (k,s,p,d,g))
                    replace_modules[name] = ub
                else : 
                    db = DownInvertibleBlock(in_c, out_c, (k,s,p,d,g))
                    replace_modules[name] = db
                   
               
    #print("after for loop", replace_modules)       
    block._modules = replace_modules
    
    
def dfs_conv2d_to_invertible(top_module, inplace=True) : 
    conv2d_to_invertible(top_module, inplace=True)
    for name, module in top_module._modules.items() : 
        if isinstance(module, memcnn.InvertibleModuleWrapper) : 
            continue
        #print(name, len(module._modules))
        if len(module._modules) > 0 : 
            dfs_conv2d_to_invertible(module)