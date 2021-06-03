import torch
import torch.nn as nn
import torch.nn.functional as F
from stitchable_conv.stitcher import Stitcher
from stitchable_conv.stitcher_conv2d_weight import Stitcher2dForWeight
from tqdm import tqdm

# ======================================================== #
# 메모리 debug용 함수
# ======================================================== #
USE_TQDM = False # stitching시에 tqdm 표시 여부
DEBUG_MEMORY = True # 모든 메모리 사용량

def debug_memory(title, level=2):
  # [torch.cuda.* 사용법] https://pytorch.org/docs/stable/cuda.html
  if DEBUG_MEMORY:
    if level == 0:
      print(f'''# ======================================================== #
# {title}
# ======================================================== #''')
    elif level == 1:
      print(f'# ---------------------------- {title} ---------------------------- #')
    elif level == 2:
      print(f'# {title}')
    print(f'  alloc={torch.cuda.memory_allocated()/1024/1024:.0f} / {torch.cuda.max_memory_allocated()/1024/1024:.0f} MB, reserved={torch.cuda.memory_reserved()/1024/1024:.0f} / {torch.cuda.max_memory_reserved()/1024/1024:.0f} MB ')

# ======================================================== #
# stitchable conv 구현
# ======================================================== #
def conv2d_weight(input, weight_size, grad_output, stride, padding):
  dilation = 1
  assert stride == 1 and padding == 0
  assert list(weight_size[2:]) == [3,3]

  in_channels = input.shape[1]
  out_channels = grad_output.shape[1]
  min_batch = input.shape[0]

  grad_output = grad_output.contiguous().repeat(
    1, in_channels, 1, 1)
  grad_output = grad_output.contiguous().view(
    grad_output.shape[0] * grad_output.shape[1], 1, grad_output.shape[2],grad_output.shape[3])

  input = input.contiguous().view(
    1, input.shape[0] * input.shape[1], input.shape[2], input.shape[3])

  grad_weight = torch.conv2d(input, grad_output, None, dilation, padding, stride,
    groups=in_channels * min_batch)

  grad_weight = grad_weight.contiguous().view(
    min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2], grad_weight.shape[3])

  return grad_weight.sum(dim=0).view(
    in_channels, out_channels, grad_weight.shape[2], grad_weight.shape[3]).transpose(0, 1).narrow(
    2, 0, weight_size[2]).narrow(3, 0, weight_size[3])

def stitchable_conv2d_weight(input, weight_size, grad_out, stride, padding, fetch_shape, device):
  assert input.device == torch.device("cpu")
  assert grad_out.device == torch.device("cpu")
  assert stride == 1 and padding == 1 # 3x3 conv, stride=1, padding=1인 경우에 대해서만 우선 생각함
  assert list(weight_size[2:]) == [3,3]

  reception_shape = [1,1]
  stw = Stitcher2dForWeight(input, grad_out, fetch_shape, reception_shape)
  debug_memory("[start] stitchable_conv2d_weight: stitching", level=2)
  grad_weight = torch.zeros(weight_size, device=device)
  for i in tqdm(range(len(stw)), desc="backward: grad_weight", disable=not USE_TQDM):
    crop_input, crop_grad = stw.get(i)
    crop_input, crop_grad = crop_input.to(device), crop_grad.to(device)
    crop_grad_weight = conv2d_weight(crop_input, weight_size, crop_grad, stride=1, padding=0)
    grad_weight += crop_grad_weight
  debug_memory("[end  ] stitchable_conv2d_weight: stitching", level=2)
  return grad_weight # on device

class StitchableConv2dFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int, padding: int,
    fetch_shape, reception_shape, grid_shape):
    assert input.device == torch.device("cpu")
    device = weight.device

    ctx.fetch_shape = fetch_shape
    ctx.reception_shape = reception_shape
    ctx.grid_shape = grid_shape
    ctx.device = device

    ctx.save_for_backward(input.detach().cpu(), weight)
    ctx.stride = stride
    ctx.padding = padding

    out_shape = torch.tensor([input.shape[0], weight.shape[0], input.shape[2], input.shape[3]])
    debug_memory("[start] StitchableConv2dFunction.forward: stitching", level=2)
    st = Stitcher(input, out_shape, fetch_shape, reception_shape, grid_shape)
    with torch.no_grad():
      for i in tqdm(range(len(st)), desc="forward", disable=not USE_TQDM):
        crop = st.get(i)
        crop = crop.to(device)
        crop_processed = F.conv2d(crop, weight, bias, stride, padding)
        st.put(i, crop_processed.cpu())
      out = st.get_stiched()
    debug_memory("[end  ] StitchableConv2dFunction.forward: stitching", level=2)
    return out
  
  @staticmethod
  def backward(ctx, grad_out: torch.Tensor):
    assert grad_out.device == torch.device("cpu")

    fetch_shape = ctx.fetch_shape
    reception_shape = ctx.reception_shape
    grid_shape = ctx.grid_shape

    input, weight = ctx.saved_tensors
    assert input.device == torch.device("cpu")
    device = weight.device
    stride = ctx.stride
    padding = ctx.padding

    # for grad_input
    debug_memory("[start] StitchableConv2dFunction.backward: stitching grad_input", level=2)
    out_shape = torch.tensor([grad_out.shape[0], weight.shape[1], grad_out.shape[2], grad_out.shape[3]])
    st = Stitcher(grad_out, out_shape, fetch_shape, reception_shape, grid_shape)
    with torch.no_grad():
      for i in tqdm(range(len(st)), desc="backward: grad_input", disable=not USE_TQDM):
        crop = st.get(i)
        crop = crop.to(device)
        crop_processed = F.conv_transpose2d(crop, weight, None, stride, padding)
        st.put(i, crop_processed.cpu())
      grad_input = st.get_stiched()
    debug_memory("[end  ] StitchableConv2dFunction.backward: stitching grad_input", level=2)

    # for grad_weight (python naive 구현)
    # with torch.no_grad(): # 이 부분은 kernel 못건드리면 이렇게 for로 구현해야함
    # 	grad_out_sum = grad_out.sum(dim=0)[None]
    # 	input_sum = input.sum(dim=0)[None]
    # 	grad_weight = torch.zeros_like(weight)
    # 	for i in range(grad_out.shape[1]):
    # 		for j in range(input.shape[1]):
    # 			grad_weight[i,j,:,:] = F.conv2d(grad_out_sum[:,i:i+1,:,:], input_sum[:,j:j+1,:,:], stride=stride, padding=padding)[0,0]

    # for grad_weight (torch.nn.grad.conv2d_weight 형태를 이용)
    # naive구현에 비하면 torch.nn.grad.conv2d_weight가 더 빠르긴함
    # https://discuss.pytorch.org/t/implementing-a-custom-convolution-using-conv2d-input-and-conv2d-weight/18556/7
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/grad.py
    # https://www.gitmemory.com/issue/pytorch/pytorch/7806/494793002
    # [cudnnConvolutionBackwardFilter] https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardFilter

    debug_memory("[start] StitchableConv2dFunction.backward: stitching grad_weight", level=2)
    with torch.no_grad():
      # grad_weight = nn.grad.conv2d_weight(input, weight.shape, grad_out, stride, padding) # vanilla conv2d_weight
      grad_weight = stitchable_conv2d_weight(input, weight.shape, grad_out, stride, padding, fetch_shape, device) # stitchable conv2d_weight
    debug_memory("[end  ] StitchableConv2dFunction.backward: stitching grad_weight", level=2)
    
    # for grad_bias
    debug_memory("[start] StitchableConv2dFunction.backward: stitching grad_bias", level=2)
    with torch.no_grad():
      grad_bias = grad_out.sum(dim=[0,2,3])
      grad_bias = grad_bias.to(device)
    debug_memory("[end  ] StitchableConv2dFunction.backward: stitching grad_bias", level=2)

    return grad_input, grad_weight, grad_bias, None, None, None, None, None

class StitchableConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding, fetch_shape):
    """
    (1) 해당 모듈은 input, output tensor는 모두 cpu tensor이다.
    (2) fetch_shape 만큼씩 gpu로 옮겨서 연산을 수행한다.
    (3) backward 시에도 fetch_shape 만큼씩 gpu로 옮겨서 연산을 수행한다.
    (4) test1, test2, test3에서 사용법을 참조
    (주의) 현재는 kernel_size=3, stride=1, padding=1인 케이스만 지원
    (주의) 현재는 python에서 grad_weight를 구하기 때문에 속도가 꽤 느림
      --> cudnn 사용해서 c++빌드후에 torch 바인딩해서 사용할 수 있음 
        
    Args:
        in_channels:
        out_channels:
        kernel_size:
        stride:
        padding:
        fetch_shape: 한번에 gpu로 연산하는 패치의 크기. [ny,nx]
    """
    super().__init__()
    # self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))
    # self.bias = nn.Parameter(torch.rand(out_channels))
    self.weight = nn.Parameter(torch.ones(out_channels, in_channels, kernel_size, kernel_size))
    self.bias = nn.Parameter(torch.zeros(out_channels))
    self.stride = stride
    self.padding = padding
    self.fetch_shape = fetch_shape
    self.reception_shape = [1,1]
    self.grid_shape = [1,1]
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
  
  def forward(self, input: torch.Tensor):
    """
    Args:
        input (torch.Tensor): cpu tensor
    Returns:
        cpu tensor
    """
    return StitchableConv2dFunction.apply(input, self.weight, self.bias, self.stride, self.padding,
      self.fetch_shape, self.reception_shape, self.grid_shape)

    def __str__(self) :
        return ("StitchableConv2d(in_channel=%d, out_channel=%d, kernel=%d, stride=%d, padding=%d, fetch_shape=" % \
                (self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)) + str(self.fetch_shape)
    
    def __repr__(self) : 
        return str(self)

class CpuToGpu(nn.Module):
  def __init__(self, device):
    super().__init__()
    self.device = device

  def forward(self, input: torch.Tensor):
    return input.to(self.device)

class GpuToCpu(nn.Module):
  def forward(self, input: torch.Tensor):
    return input.to("cpu")