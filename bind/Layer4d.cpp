#include "Layer4d.h"
using namespace std;

#define DEBUG_MEMORY 0
#define DEBUG_WARNING 0
#define DEBUG_DIM 0
#define CUDNN_DATA_REAL CUDNN_DATA_FLOAT
#define REALLOC_MEMORY 0

Layer4d::Layer4d() {
	init();
}
Layer4d::Layer4d(bool alloc) {
	constr(alloc);
}
void Layer4d::init() {
	cudnn_.init();
	cublas_.init();
	desc_relu_.init();
	mem_workspace_.init();
}

void Layer4d::constr(bool alloc) {
	init();
	if (alloc) {
		cudnn_ = HCudnnHandle(true);
		cublas_ = HCublasHandle(true);
		desc_relu_ = HCudnnActivationDesc(true);
		mem_workspace_ = HCudaMemory<uint8_t>(1024 * 1024);
	}
}

const HCudnnHandle& Layer4d::getCudnn() const {
	return cudnn_;
}

const HCublasHandle& Layer4d::getCublas() const {
	return cublas_;
}

#if 0 // cudnn legacy version
void Layer4d::conv2d(
	Tensor4d& out,
	const Tensor4d& in,
	const Weight4d& weight,
	const Bias4d& bias,
	const std::vector<int>& strides,
	const std::vector<int>& padding
) {
#if DEBUG_MEMORY
	if (strides.size() != 2) {
		_raise("Layer4d::conv2d: strides.size != 2");
	}
	if (padding.size() != 2) {
		_raise("Layer4d::conv2d: padding.size != 2");
	}
#endif
	float one = 1;
	float zero = 0;

	// set desc conv
	HCudnnConvDesc desc_conv(padding, strides, { 1,1 }); // pad, strides, dilation

	// get output dim
	vector<int> dim_out(4); // b,c,y,x
	cuda(cudnnGetConvolutionNdForwardOutputDim(
		desc_conv.data(),
		in.desc().data(),
		weight.desc().data(),
		4, dim_out.data()
	));

	// out 디멘젼이 다르면
	if (dim_out != out.desc().dims()) {
		int numel = VectorHelper::prod(dim_out);
#if REALLOC_MEMORY
		// 메모리가 더 작다면 다시 만듬
		if (numel > out.mem().size()) {
			out = Tensor4d(HCudaMemory<float>(numel), HCudnnTensorDesc(dim_out));
		}
		// 충분하면 메모리는 재사용
		else {
			out = Tensor4d(out.mem(), HCudnnTensorDesc(dim_out));
		}
#else // 메모리가 항상 충분하다 가정
		out = Tensor4d(out.mem(), HCudnnTensorDesc(dim_out));
#endif
#if DEBUG_MEMORY
		// 메모리가 작다면 예외
		if (numel > out.mem().size()) {
			_raise("Layer4d::conv2d: out.mem is not enough");
		}
#endif	
	}

	// conv algo 찾음
	cudnnConvolutionFwdAlgo_t algo_conv;
	cuda(cudnnGetConvolutionForwardAlgorithm(
		cudnn_.data(), in.desc().data(), weight.desc().data(), desc_conv.data(), out.desc().data(),
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_conv
	));

	// 필요한 workspace 크기 구함
	size_t sizeInBytes;
	cuda(cudnnGetConvolutionForwardWorkspaceSize(
		cudnn_.data(), in.desc().data(), weight.desc().data(), desc_conv.data(), out.desc().data(),
		algo_conv, &sizeInBytes
	));

	// workspace가 더 작다면 늘려줌
	if (mem_workspace_.size() < sizeInBytes) {
		mem_workspace_ = HCudaMemory<uint8_t>(sizeInBytes); // 이것 때문에 const method 안됨
	}

	// conv run
	cuda(cudnnConvolutionForward(
		cudnn_.data(),
		&one, in.desc().data(), in.mem().data(),
		weight.desc().data(), weight.mem().data(),
		desc_conv.data(), algo_conv, mem_workspace_.data(), mem_workspace_.size(),
		&zero, out.desc().data(), out.mem().data()
	));

	// add b
	cuda(cudnnAddTensor(
		cudnn_.data(),
		&one, bias.desc().data(), bias.mem().data(),
		&one, out.desc().data(), out.mem().data()
	));
}
#endif

void Layer4d::add(
	Tensor4d& out,
	const Tensor4d& in
) const {
	float one = 1;
	cuda(cudnnAddTensor(
		cudnn_.data(),
		&one, in.desc().data(), in.mem().data(),
		&one, out.desc().data(), out.mem().data()
	));
}

void Layer4d::add(
	Tensor4d& out,
	const Tensor4d& in1,
	const Tensor4d& in2
) const {
	float one = 1;
	copy(out, in1);
	cuda(cudnnAddTensor(
		cudnn_.data(),
		&one, in2.desc().data(), in2.mem().data(),
		&one, out.desc().data(), out.mem().data()
	));
}

void Layer4d::copy(
	Tensor4d& out,
	const Tensor4d& in
) const {
	out = Tensor4d(HCudaMemory<float>(in.mem().size()), HCudnnTensorDesc(in.desc().dims()));
	cuda(cudaMemcpy(out.mem().data(), in.mem().data(), in.desc().sizeByDims() * sizeof(float), cudaMemcpyDeviceToDevice));
}