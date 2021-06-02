import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from stitchable_conv.stitcher import mrange

class AutoPadding2d(nn.Module):
  def __init__(self, reception_shape):
    super().__init__()
    self.reception_shape = torch.tensor(reception_shape, dtype=torch.int64)

  def forward(self, input, slices: typing.List[slice]):
    shape = torch.tensor(input.shape, dtype=torch.int64)
    nb,nc,ny,nx = shape
    sb,sc,sy,sx = slices

    pady0 = max(0 - sy.start, 0)
    pady1 = max(sy.stop - ny, 0)
    padx0 = max(0 - sx.start, 0)
    padx1 = max(sx.stop - nx, 0)
  
    if pady0 != 0 or pady1 != 0 or padx0 != 0 or padx1 != 0:
      input = F.pad(input, [padx0, padx1, pady0, pady1], mode="constant")
      sy = slice(sy.start + pady0, sy.stop + pady0, sy.step)
      sx = slice(sx.start + padx0, sx.stop + padx0, sx.step) 
      return input[sb, sc, sy, sx]
    else:
      return input[sb, sc, sy, sx]
  
class Stitcher2dForWeight:
  def __init__(self, input: torch.Tensor, grad: torch.Tensor, fetch_shape, reception_shape):
    assert input.shape[2:] == grad.shape[2:] # input, output의 spatial dim이 같은 conv에 대해서만 고려
    self.ap = AutoPadding2d(reception_shape)
    self.input = input
    self.grad = grad

    shape = torch.tensor(input.shape[2:], dtype=torch.int64)
    fetch_shape = torch.tensor(fetch_shape, dtype=torch.int64)
    reception_shape = torch.tensor(reception_shape, dtype=torch.int64)
    valid_shape = fetch_shape - 2*reception_shape

    assert torch.all(valid_shape > 0), "fetch_shape too small"

    input = F.pad(input, [reception_shape[1], reception_shape[1], reception_shape[0], reception_shape[0]], mode="constant")

    self.fetch_list_input = []
    self.fetch_list_grad = []

    ranges = [range(0, shape[i], valid_shape[i]) for i in range(2)]
    ny, nx = shape
    vy, vx = valid_shape
    ry, rx = reception_shape
    for es in mrange(*ranges):
      iy, ix = es
      self.fetch_list_input.append([slice(iy-ry, min(iy+vy+ry, ny+ry)), slice(ix-rx, min(ix+vx+rx, nx+rx))])
      self.fetch_list_grad.append([slice(iy, min(iy+vy, ny)), slice(ix, min(ix+vx, nx))])
  
  def __len__(self):
    return len(self.fetch_list_input)

  def get(self, i):
    # crop_input = self.input.__getitem__(slice(None), slice(None), self.fetch_list_input[i])
    # crop_grad = self.grad.__getitem__(slice(None), slice(None), self.fetch_list_grad[i])
    crop_input = self.ap(self.input, [slice(None), slice(None), *self.fetch_list_input[i]])
    crop_grad = self.ap(self.grad, [slice(None), slice(None), *self.fetch_list_grad[i]])
    return crop_input, crop_grad

if __name__ == "__main__":
  input = torch.arange(49).reshape(1,1,7,7).float()
  grad = torch.arange(49).reshape(1,1,7,7).float()
  stw = Stitcher2dForWeight(input, grad, fetch_shape=[4,4], reception_shape=[1,1])
  for i in range(len(stw)):
    crop_input, crop_grad = stw.get(i)
    print(f'*** {i} ***')
    print(f'crop_input.shape = {crop_input.shape}')
    print(f'crop_grad.shape = {crop_grad.shape}')

    print(crop_input)
    print(crop_grad)
