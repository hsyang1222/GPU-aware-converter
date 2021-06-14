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
  
  def forward(self, input: torch.Tensor):
    """
    Args:
        input (torch.Tensor): cpu tensor
    Returns:
        cpu tensor
    """
    return StitchableConv2dFunction.apply(input, self.weight, self.bias, self.stride, self.padding,
      self.fetch_shape, self.reception_shape, self.grid_shape)

class CpuToGpu(nn.Module):
  def __init__(self, device):
    super().__init__()
    self.device = device

  def forward(self, input: torch.Tensor):
    return input.to(self.device)

class GpuToCpu(nn.Module):
  def forward(self, input: torch.Tensor):
    return input.to("cpu")

# ======================================================== #
# unit test
# ======================================================== #
def test0(): # gpu, cpu 메모리 사이를 왔다 갔다해도 backward가 잘되는지 확인
  gpu2cpu = GpuToCpu()
  cpu2gpu = CpuToGpu(torch.device("cuda:0"))
  input = torch.ones(1,1,3,3, device=torch.device("cuda:0"), requires_grad=True)
  out = input
  out = gpu2cpu(out)
  out = cpu2gpu(out)
  out.sum().backward()
  print(input.grad)

def test1(): # stitchable conv가 vanila conv와 동작이 같은지 작은 이미지로 직접 숫자 확인
  # ---------------------------- stitchable conv2d ---------------------------- #
  model = StitchableConv2d(1,1,3,1,1, fetch_shape=[5,5])
  model.weight.data = torch.arange(36).reshape(2,2,3,3).float()
  model.bias.data = torch.ones(2)
  input = torch.ones(1,2,10,10, requires_grad=True)
  out = model(input)			
  out.sum().backward()
  print("*** stitchable conv2d ***")
  print(input.grad)
  print(model.weight.grad)
  print(model.bias.grad)

  # ---------------------------- vanila conv2d ---------------------------- #
  model2 = nn.Conv2d(1,1,3,1,1)
  model2.weight.data = torch.arange(36).reshape(2,2,3,3).float() # model2.weight.data = torch.ones(1,1,3,3)
  model2.bias.data = torch.ones(2)
  input = torch.ones(1,2,10,10, requires_grad=True)
  out = model2(input)			
  out.sum().backward()
  print("*** vanila conv2d ***")
  print(input.grad)
  print(model2.weight.grad)
  print(model2.bias.grad)

def test2(): # stitchable conv가 random 숫자에 대해서도 vanila conv와 동작이 같은지 확인
  input = torch.rand(1,2,64,64)
  weight = torch.rand(2,2,3,3)
  bias = torch.rand(2)

  # ---------------------------- stitchable conv2d ---------------------------- #
  model1 = StitchableConv2d(1,1,3,1,1, fetch_shape=[5,5])
  model1.weight.data = weight.clone()
  model1.bias.data = bias.clone()
  input1 = input.clone().requires_grad_(True)
  out1 = model1(input1)
  out1.sum().backward()
  print("*** stitchable conv2d ***")
  print(input1.grad)
  print(model1.weight.grad)
  print(model1.bias.grad)

  # ---------------------------- vanila conv2d ---------------------------- #
  model2 = nn.Conv2d(1,1,3,1,1)
  model2.weight.data = weight.clone()
  model2.bias.data = bias.clone()
  input2 = input.clone().requires_grad_(True)
  out2 = model2(input2)			
  out2.sum().backward()
  print("*** vanila conv2d ***")
  print(input2.grad)
  print(model2.weight.grad)
  print(model2.bias.grad)

  print(f'out err = {(out1-out2).abs().mean()}')
  print(f'input.grad err = {(input1.grad-input2.grad).abs().mean()}')
  print(f'weight.grad err = {(model1.weight.grad - model2.weight.grad).abs().mean()}')
  print(f'bias.grad err = {(model1.bias.grad - model2.bias.grad).abs().mean()}')

def test3(): # stitchable conv가 큰 이미지에 대해서도 잘 동작하는지 확인
  # 3080 (11GB) 기준에서 1,64,4096,4096을 수행하면 model2는 터지지만 model1은 수행가능한것 확인함
  # 다만, 속도 이슈가 있음. 현재 grad_weight를 구하는 함수인 conv2d_weight는 pytorch 구현을 따라서 속도가 매우 느림.
  # 속도를 올리려면 cuda kernel을 직접 건드려야 하는 상황
  
  input = torch.rand(1,64,4096,4096) # 1,64,2048,2048 에서 동작하면 vanila conv도 메모리 안터지고 동작
  weight = torch.rand(64,64,3,3)
  bias = torch.rand(64)

  # ---------------------------- stitchable conv2d ---------------------------- #
  input1 = input.clone().requires_grad_(True)
  model1 = StitchableConv2d(64,64,3,1,1, fetch_shape=[512,512]).cuda()
  model1.weight.data = weight.clone().cuda()
  model1.bias.data = bias.clone().cuda()
  for i in tqdm(range(1)):
    out1 = model1(input1)
    out1.sum().backward()

  # ---------------------------- vanila conv2d ---------------------------- #
  input2 = input.clone().cuda().requires_grad_(True)
  model2 = nn.Conv2d(64,64,3,1,1).cuda()
  model2.weight.data = weight.clone().cuda()
  model2.bias.data = bias.clone().cuda()
  for i in tqdm(range(1)):
    out2 = model2(input2)
    out2.sum().backward()

  with torch.no_grad():
    print(f'out err = {(out1.cpu()-out2.cpu()).abs().mean()}')
    print(f'input.grad err = {(input1.grad.cpu()-input2.grad.cpu()).abs().mean()}')
    print(f'weight.grad err = {(model1.weight.grad.cpu() - model2.weight.grad.cpu()).abs().mean()}')
    print(f'bias.grad err = {(model1.bias.grad.cpu() - model2.bias.grad.cpu()).abs().mean()}')

def test4(): # unet에서 stitcable conv 사용법 예시. vanilla unet과 stitchable unet의 메모리 사용량도 확인함
  class Model(nn.Module):
    def __init__(self, stitchable=True):
      super().__init__()
      if stitchable:
        self.enc0 = StitchableConv2d(1,64,3,1,1,[128,128])
        self.enc1 = StitchableConv2d(64,64,3,1,1,[128,128])
        self.enc2 = StitchableConv2d(64,64,3,1,1,[128,128])
        self.dec1 = StitchableConv2d(128,64,3,1,1,[128,128])
        self.dec0 = StitchableConv2d(128,1,3,1,1,[128,128])
      else:
        self.enc0 = nn.Conv2d(1,64,3,1,1)
        self.enc1 = nn.Conv2d(64,64,3,1,1)
        self.enc2 = nn.Conv2d(64,64,3,1,1)
        self.dec1 = nn.Conv2d(128,64,3,1,1)
        self.dec0 = nn.Conv2d(128,1,3,1,1)
      
    def forward(self, input):
      out = input
      # ---------------------------- level 0 ---------------------------- #
      debug_memory("enc0 forward", level=1)
      out = F.relu(self.enc0(out))
      out0 = out

      # ---------------------------- level 1 ---------------------------- #
      debug_memory("enc1 forward", level=1)
      out = F.max_pool2d(out, 2)
      out = F.relu(self.enc1(out))
      out1 = out

      # ---------------------------- level 2 ---------------------------- #
      debug_memory("enc2 forward", level=1)
      out = F.max_pool2d(out, 2)
      out = F.relu(self.enc2(out))

      # ---------------------------- level 1 ---------------------------- #
      debug_memory("dec1 forward", level=1)
      out = F.upsample_nearest(out, scale_factor=2)
      out = torch.cat([out, out1], dim=1)
      out = F.relu(self.dec1(out))

      # ---------------------------- level 0 ---------------------------- #
      debug_memory("dec0 forward", level=1)
      out = F.upsample_nearest(out, scale_factor=2)
      out = torch.cat([out, out0], dim=1)
      out = self.dec0(out)

      debug_memory("end forward", level=1)
      return out
  # ---------------------------- case 1: normal unet ---------------------------- #
  def test_non_stitchable():
    def scope(): # reference cnt를 줄이기위한 scope
      debug_memory("before init", level=0)
      model = Model(False).cuda()
      print(model)
      for m in model.modules():
        m.register_full_backward_hook(lambda a, b, c: debug_memory("[backward] ", level=2))
      input = torch.ones(1,1,1024,1024).cuda()
      label = torch.zeros(1,1,1024,1024).cuda()
      debug_memory("after init", level=0)
      out = model(input)
      debug_memory("after forward", level=0)
      loss = F.mse_loss(out, label)
      loss.backward()
      debug_memory("after backward", level=0)
      torch.cuda.empty_cache()
    scope()
    debug_memory("after free", level=0)

  # ---------------------------- case 2: stitchable unet ---------------------------- #
  def test_stitchable():
    def scope():
      debug_memory("before init", level=0)
      model = Model(True).cuda()
      for m in model.modules():
        m.register_full_backward_hook(lambda a, b, c: debug_memory("[backward] ", level=2))
      input = torch.ones(1,1,1024,1024) # must be cpu tensor
      label = torch.zeros(1,1,1024,1024)
      debug_memory("after init", level=0)
      out = model(input)
      debug_memory("after forward", level=0)
      loss = F.mse_loss(out, label)
      loss.backward()
      debug_memory("after backward", level=0)
      torch.cuda.empty_cache()
    scope()
    debug_memory("after free", level=0)

  print(f'''# ******************************************************** #
# test_non_stitchable
# ******************************************************** #''')
  test_non_stitchable()

  torch.cuda.reset_max_memory_allocated()
  # torch.cuda.reset_max_memory_cached()
  print("\n\n\n\n")

  print(f'''# ******************************************************** #
# test_stitchable
# ******************************************************** #''')
  test_stitchable()

def test5(): # torch에서 gpu memory 사용량을 low level하게 체크하는 방식을 보여주는 예제
  # [torch.cuda.* 사용법] https://pytorch.org/docs/stable/cuda.html
  def print_memory(title):
    print(f'# ---------------------------- {title} ---------------------------- #')
    # 아래의 메모리 사용법은 torch가 구현해둔 cuda kernel 함수들이 차지하는 메모리 (800~1000MB) 정도는 
    # 전혀 반영하지 않은 메모리 사용량임을 유의. after burner, nvidia-smi로 확인한 GPU 메모리 사용량은
    # torch의 cuda kernel 함수들이 차지하는 메모리 (800~1000MB)도 반영되어있음

    # [memory_allocated] 현재 tensor가 실제로 먹은 메모리. python에서 객체에 대한 pointer가 refernce cnt를 
    # 관리하는 방식에 따라 free된 것으로 취급해서 사용량을 표시함.
    alloc = torch.cuda.memory_allocated()/1024/1024
    # [max_memory_allocated] 이전 history중에서 alloc의 최대값. 
    # torch.cuda.reset_max_memory_allocated()로 초기화 가능
    max_alloc = torch.cuda.max_memory_allocated()/1024/1024
    # [memory_reserved] 한번이라도 먹은 메모리는 torch에서 관리함.
    # 이때, torch가 관리하고 있는 메모리의 총량을 표시함
    reserved = torch.cuda.memory_reserved()/1024/1024
    # [max_memory_reserved] 이전 history중 reserved의 최대값.
    # torch.cuda.reset_max_memory_cached()로 초기화 가능
    max_reserved = torch.cuda.max_memory_reserved()/1024/1024
    print(f'alloc={alloc:.0f} / {max_alloc:.0f} MB, reserved={reserved:.0f} / {max_reserved:.0f} MB ')

  print_memory("init")
  torch.ones(3).cuda() # torch가 최초로 cuda 사용시, torch의 cuda kernel 함수들이 로드됨 (800~1000MB 정도 차지함)
  print_memory("cuda on")
  def gpu_memory_chk_example():
    print_memory("before alloc")
    a = torch.ones(1024//4,1024,1024).cuda() # 실제 객체를 mem_A라 부르면
    b = a
    del a # a에 대한 reference cnt만 감소. b가 여전히 mem_A를 포인팅 하므로 mem_A는 사라지지 않음
    print_memory("after alloc")
  gpu_memory_chk_example() 
  # 함수가 끝났으므로 mem_A를 가르키는 모든 포인터가 사라짐. 
  # reference cnt=0이므로 allocated에서는 제외됨
  # 단, reserved에서는 사라지지 않음
  print_memory("after free")
  torch.cuda.empty_cache() 
  # empty_cache 후에는 pytorch가 괸리하는 gpu memory를 모두 해제함. reserved도 0이 됨.
  # 단, torch의 cuda kernel 함수들이 차지하는 메모리 (800~1000MB)는 프로그램이 종료될때까지 절대 해제되지 않음.
  print_memory("after torch.cuda.empty_cache")
  print("end program")

# ======================================================== #
# unit test main
# ======================================================== #
if __name__ == "__main__":
  # test0() # gpu, cpu 메모리 사이를 왔다 갔다해도 backward가 잘되는지 확인
  # test1() # stitchable conv가 vanila conv와 동작이 같은지 작은 이미지로 직접 숫자 확인
  # test2() # stitchable conv가 random 숫자에 대해서도 vanila conv와 동작이 같은지 확인
  # test3() # stitchable conv가 큰 이미지에 대해서도 잘 동작하는지 확인
  test4() # unet에서 stitcable conv 사용법 예시. vanilla unet과 stitchable unet의 메모리 사용량도 확인함
  # test5() # torch에서 gpu memory 사용량을 low level하게 체크하는 방식을 보여주는 예제
  pass

#%% # ==================================================== #
# debug
# ======================================================== #
# https://discuss.pytorch.org/t/implementing-a-custom-convolution-using-conv2d-input-and-conv2d-weight/18556/7
# https://github.com/pytorch/pytorch/blob/master/torch/nn/grad.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm
# x = torch.ones(1,64,1024,1024).cuda()
# y = torch.ones(1,64,1024,1024).cuda()
# weight = torch.ones(64,64,3,3).cuda()
# stride = 1
# padding = 1
# # for i in tqdm(range(64*64)):
# # 	F.conv2d(x,y,stride=1,padding=1)

# for i in tqdm(range(100)):
# 	out = nn.grad.conv2d_weight(x, torch.ones(1,1,3,3).shape, y, stride, padding)
# 	out = out.cpu()

# for i in tqdm(range(100)):
# 	out = nn.grad.conv2d_input(x.shape, weight, y, stride, padding)
# 	out = out.cpu()

# for i in tqdm(range(100)): # nn.grad.conv2d_input과 속도 같음
# 	out = F.conv_transpose2d(x, weight, None, stride, padding)
# 	out = out.cpu()

# ======================================================== #
# 미팅 논의 내용
# ======================================================== #
# unet
# conv2d

# b,c,y,x
# 1,64,4096,4096

# ch=1 --> ch=64 --> ch=64 --> ch=64 --> ch=1

## conv2d, kernel=3x3, stride=1, padding=1

#(1) cpu
# maxpool: stitcahble: forward / backward
# relu: stitcahble
# upsample: nearest upsample

#(2)
# weight grad binding

#(3)
# model에서 conv2d, maxpool, relu --> stitchable로 자동 변환