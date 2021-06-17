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
    input1 = input.clone().requires_grad_(True)
    model1 = StitchableConv2d(64,64,3,1,1, fetch_shape=[512,512]).cuda()
    model1.weight.data = weight.clone().cuda()
    model1.bias.data = bias.clone().cuda()
    out1 = model1(input1)
    out1.sum().backward()

def test_non_stitch():
    input = torch.rand(1,64,4096,4096)
    weight = torch.rand(64,64,3,3)
    bias = torch.rand(64)

    # ---------------------------- vanila conv2d ---------------------------- #
    input2 = input.clone().cuda().requires_grad_(True)
    model2 = nn.Conv2d(64,64,3,1,1).cuda()
    model2.weight.data = weight.clone().cuda()
    model2.bias.data = bias.clone().cuda()
    out2 = model2(input2)
    out2.sum().backward()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_stitch", type=int, default=0)
    args = parser.parse_args()
    if args.use_stitch:
        test_stitch()
    else:
        test_non_stitch()