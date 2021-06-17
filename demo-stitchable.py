import torch
import torch.nn as nn
import torch.nn.functional as F
from stitchable_conv.StitchableConv2d import StitchableConv2d
import argparse

def test_stitch():
    input = torch.rand(1,64,4096,4096)
    weight = torch.rand(64,64,3,3)
    bias = torch.rand(64)

    # ---------------------------- stitchable conv2d ---------------------------- #
    print("start")
    input = input.clone().requires_grad_(True)
    model = StitchableConv2d(64,64,3,1,1, fetch_shape=[512,512], use_tqdm=True).cuda()
    model.weight.data = weight.clone().cuda()
    model.bias.data = bias.clone().cuda()
    out = model(input)
    out.sum().backward()
    print("end")

def test_non_stitch():
    input = torch.rand(1,64,4096,4096)
    weight = torch.rand(64,64,3,3)
    bias = torch.rand(64)

    # ---------------------------- vanila conv2d ---------------------------- #
    print("start")
    input = input.clone().cuda().requires_grad_(True)
    model = nn.Conv2d(64,64,3,1,1).cuda()
    model.weight.data = weight.clone().cuda()
    model.bias.data = bias.clone().cuda()
    out = model(input)
    out.sum().backward()
    print("end")

if __name__ == "__main__":
    #
    # python demo-stitchable.py --use_stitch 1 # working
    # python demo-stitchable.py --use_stitch 0 # OOM

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_stitch", type=int, default=0)
    args = parser.parse_args()
    if args.use_stitch:
        test_stitch()
    else:
        test_non_stitch()