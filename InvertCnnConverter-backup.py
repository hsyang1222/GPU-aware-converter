import torch
import torch.nn as nn
import memcnn
from collections import OrderedDict
import copy

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
               #print("input dim and output dim is not matched")
                pass
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