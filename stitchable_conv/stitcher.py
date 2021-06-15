import numpy as np
import torch
import torch.nn.functional as F

# torch.set_printoptions(linewidth=100)
# ---------------------------- utils ---------------------------- #
def ceil(norminator, denorminator):
  return (norminator - 1) // denorminator + 1

def floor(norminator, denorminator):
  return norminator // denorminator

def gridfy(size, grid, grow=True):
  if grow:
    return ceil(size, grid) * grid
  else:
    return floor(size, grid) * grid

def gridfy_padding(img, grid_shape):
  shape = torch.tensor(img.shape, dtype=torch.int64)
  grid_shape = torch.tensor(grid_shape, dtype=torch.int64)
  pad_shape = gridfy(shape, grid_shape) - shape
  return F.pad(img, [0, pad_shape[1], 0, pad_shape[0]], mode="reflect")

def mrange(*iterables):
  if not iterables:
    yield []
  else:
    for item in iterables[0]:
      for rest_tuple in mrange(*iterables[1:]):
        yield [item] + rest_tuple

# ---------------------------- sticher ---------------------------- #
class Stitcher:
  def __init__(self, whole, out_shape, fetch_shape, reception_shape, grid_shape, grow_fetch_shape=False):
    """
    Example:
      sticher = Stitcher(whole, out_shape, fetch_shape, reception_shape, grid_shape)
      for i in range(len(sticher)):
        crop = sticher.get(i) # (grid_shape의 배수가 되도록 확장된) fetch_shape
        crop_processed = net(crop) # nn module
        sticher.put(i, crop_processed) # crop_processed 를 등록
      out = sticher.get_stiched() # crop_processed 를 모두 합친 결과를 리턴
      
    Args:
        whole: torch.Tensor, 임의의 shape, 전체 데이터
        out_shape: 최종 shape
        fetch_shape: .get() 으로 얻을 데이터의 크기
        reception_shape:
          (1) reception field를 제외한 크기 만큼만 stiched output을 만드는데 활용함
          (2) reception field를 고려해서 reflection padding 해서 인풋 만듬 (모서리 부분도 포함)
        grid_shape: crop_shape를 넣었지만 각 axis별로 인풋 크기가 grid_shape의 배수가 되도록 함
        grow_fetch_shape: crop_shape를 gridfy할때 크기를 늘릴지, 줄일지 결정 (True: 늘림, False: 줄임)
    """
    assert whole.ndim-2 == len(fetch_shape)
    assert whole.ndim-2 == len(reception_shape)
    assert whole.ndim-2 == len(grid_shape)

    # torch.tensor 로 만듬
    fetch_shape = torch.tensor(fetch_shape, dtype=torch.int64)
    reception_shape = torch.tensor(reception_shape, dtype=torch.int64)
    grid_shape = torch.tensor(grid_shape, dtype=torch.int64)
    whole_shape_orig = torch.tensor(whole.shape[2:], dtype=torch.int64)
    ndim = len(whole_shape_orig)

    # whole 데이터 gridfy (reflect padding)
    pad_shape = gridfy(whole_shape_orig, grid_shape, grow=True) - whole_shape_orig
    whole = F.pad(whole, [0, pad_shape[1], 0, pad_shape[0]], mode="reflect")
    shape = torch.tensor(whole.shape[2:], dtype=torch.int64)

    # 들고올 crop 사이즈 gridfy
    fetch_shape = gridfy(fetch_shape, grid_shape, grow=grow_fetch_shape)

    # reception field 크기 gridfy
    reception_shape = gridfy(reception_shape, grid_shape, grow=True)

    # valid 한 영역의 크기
    valid_shape = fetch_shape - 2*reception_shape

    # self에 등록
    self.whole = whole  # 인풋
    self.whole_shape_orig = whole_shape_orig
    self.out = torch.zeros(*out_shape, device=whole.device)  # 결과
    self.fetch_list = []
    self.valid_list = []
    self.apply_list = []

    # print(fetch_shape, reception_shape, valid_shape)

    assert torch.all(valid_shape > 0), "fetch_shape too small"

    # fetch, valid, apply list 만들기
    # ranges = [range(0, shape[i], valid_shape[i]) for i in range(ndim)]  # 각 axis 마다 range 등록
    ranges = []
    for i in range(ndim):
      range_each = []
      end = shape[i]
      v = valid_shape[i]
      r = reception_shape[i]
      for e in range(0, end, valid_shape[i]):
        range_each.append(e)
        # 마지막에 reception 버릴 필요 없는 구간을 만나면 append 후 바로 끝
        # end > e and e <= e+v+r 과 같음 (e < end 는 range를 만드는 것에서 이미 조건만족)
        if end-v-r <= e: 
          break
        # 처음에 시작하는데, 크기가 v+2*r 이내에 들어오는 경우
        # range에는 e = 0 하나만 넣어주고 break
        if e == 0 and end <= v+2*r: 
          break 
      ranges.append(range_each)

    for es in mrange(*ranges):  # ndim개의 for문, 아래 내용은 가장 내부의 for문이고 es는 ndim개의 인덱스
      fetch = [None] * ndim  # 각 axis 마다 가져올 위치
      apply = [None] * ndim  # 각 axis 마다 최종 결과에 붙여넣을 위치
      valid = [None] * ndim  # 각 axis 마다 부분 결과에서 사용할 위치
      for i in range(ndim):
        e = es[i]  # 현재 iter 커서가 머무르는 위치
        v = valid_shape[i]  # valid 한 크기
        r = reception_shape[i]  # reception field 크기
        g = grid_shape[i]  # gridfy 크기
        end = shape[i]  # 해당 axis의 크기
        if end <= v + 2*r: # 크기가 작은 경우: 통 실행
          fetch[i] = slice(0, end)
          apply[i] = slice(0, end)
          valid[i] = slice(0, end)
        elif e == 0:  # 왼쪽 끝
          fetch[i] = slice(0, v + r)
          apply[i] = slice(0, v)
          valid[i] = slice(0, v)
        elif end > e and end <= e+v+r:  # 오른쪽 끝
          fetch[i] = slice(e - r, end)
          apply[i] = slice(e, end)
          valid[i] = slice(r, end - e + r)
        else:  # 중간
          fetch[i] = slice(e - r, e + v + r)
          apply[i] = slice(e, e + v)
          valid[i] = slice(r, v + r)
      self.fetch_list.append(fetch)
      self.apply_list.append(apply)
      self.valid_list.append(valid)

  def __len__(self):
    return len(self.fetch_list)

  def get(self, i):
    # return self.whole[:,:,self.fetch_list[i]]
    return self.whole.__getitem__([slice(None), slice(None), *self.fetch_list[i]])

  def put(self, i, input):
    # self.out[:,:,self.apply_list[i]] = input[:,:,self.valid_list[i]]
    self.out.__setitem__([slice(None), slice(None), *self.apply_list[i]], input.__getitem__([slice(None), slice(None), *self.valid_list[i]]))

  def get_stiched(self):
    slices = [slice(s) for s in self.whole_shape_orig]
    # return self.out[:,:,slices]
    return self.out.__getitem__([slice(None), slice(None), *slices])

# ---------------------------- unit test ---------------------------- #
def test_stitch(shape, fetch_shape, reception_shape, grid_shape, verbose=False):
  print(f"""# ========================================================
# test_stitch(
# 	shape={shape},
# 	fetch_shape={fetch_shape},
# 	receptino_shape={reception_shape},
# 	grid_shape={grid_shape},
# 	verbose={verbose}
# )
# ========================================================""")
  img = torch.zeros(1,1,*shape)
  out_shape = img.shape
  st = Stitcher(img, out_shape, fetch_shape, reception_shape, grid_shape, grow_fetch_shape=False)
  for i in range(len(st)):
    print("************* {} ***************".format(i))
    crop = st.get(i)
    crop_processed = crop + i + 1
    print(f"crop.shape={crop_processed.shape}")
    print(f"fetch={st.fetch_list[i]}")
    print(f"valid={st.valid_list[i]}")
    print(f"apply={st.apply_list[i]}")
    st.put(i, crop_processed)
    if verbose:
      print(st.out)
  out = st.get_stiched()
  if verbose:
    print("************* result *************")
    print(out)

if __name__ == "__main__":
  test_stitch([18,18], [8,8], [1,1], [2,2], verbose=True) # 2d
  # test_stitch([3,3], [5,5], [1,1], [1,1], verbose=True) # 2d small
  # test_stitch([1,10,10], [1,5,5], [0,1,1], [1,1,1], verbose=True) # 3d (z는 1인 경우)
  # test_stitch([3,10,10], [3,5,5], [1,1,1], [1,1,1], verbose=True) # 3d
  # test_stitch([1,150,396,1133,1], [1,48,256,256,1], [0,16,48,48,0], [1,8,8,8,1], verbose=False) # 5d
  pass

# ---------------------------- scratch ---------------------------- #
