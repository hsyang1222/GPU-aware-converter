#undef slots
#include <torch/extension.h>
#include "cudnn_handle.h"
#include "memory_handle.h"
#include <dlfcn.h>

#include "cuda_global_setting.h"
#include "cudnn_handle.h"
#include "cout_overloading.h"
#include <cublas_v2.h>
#include <iostream>
#include <cudnn.h>
#include <ATen/cudnn/Handle.h>
#include <torch/nn/functional/conv.h>
// #include <cudnn_ops_infer.h>

// #include <ATen/cuda/CUDAConfig.h>
// // #include <ATen/native/cudnn/Macros.h>
// #include <ATen/ATen.h>
// #include <ATen/NativeFunctions.h>
// #include <ATen/Config.h>
// #include <ATen/cuda/Exceptions.h>
// // #include <ATen/native/cudnn/ConvShared.h>
// #include <THC/THC.h>
// #include <ATen/cudnn/Types.h>
// #include <ATen/cudnn/Utils.h>
// // #include <ATen/native/utils/ParamsHash.h>
// #include <ATen/TensorUtils.h>

// #include <cudnn_ops_infer.h>

using namespace std;

torch::Tensor f(
	torch::Tensor out,
	torch::Tensor in
) {
	py::print("run f");
	return out + in;
}

void conv2d(
	torch::Tensor out,
	torch::Tensor input,
	torch::Tensor weight
) {
	// cout << "start" << endl;
	auto sout = out.sizes();
	auto sin = input.sizes();
	auto sw = weight.sizes();

	// cout << "dumy" << endl;
	// torch::ones({1,1,10,10}).to(torch::Device(torch::DeviceType::CUDA, 0));

	// cout << "copy" << endl;
	out = out.to(torch::Device(torch::DeviceType::CUDA, 0));
	input = input.to(torch::Device(torch::DeviceType::CUDA, 0));
	weight = weight.to(torch::Device(torch::DeviceType::CUDA, 0));

	// cout << "start" << endl;
	// auto handle = HCudnnHandle(true);
	cudnnHandle_t handle = at::native::getCudnnHandle();
	// cout << "handle: " << handle << endl;

	float one = 1;
	float zero = 0;
	// cout << "desc" << endl;
	HCudnnTensorDesc out_desc({(int)sout[0], (int)sout[1], (int)sout[2], (int)sout[3]}, CUDNN_DATA_FLOAT);
	HCudnnTensorDesc input_desc({(int)sin[0], (int)sin[1], (int)sin[2], (int)sin[3]}, CUDNN_DATA_FLOAT);
	HCudnnFilterDesc weight_desc({(int)sw[0], (int)sw[1], (int)sw[2], (int)sw[3]}, CUDNN_DATA_FLOAT);
	HCudnnConvDesc conv_desc({1,1},{1,1},{1,1});

	cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

	// cout << "workspace" << endl;
	// HCudaMemory<uint8_t> workspace(1024*1024*1024);

	// cout << "run" << endl;
	cuda(cudnnConvolutionForward(
		handle, 
		&one, input_desc.data(), input.data_ptr<float>(),
		weight_desc.data(), weight.data_ptr<float>(),
		conv_desc.data(), algo, 
		// workspace.data(), workspace.size(),
		nullptr, 0,
		&zero, out_desc.data(), out.data_ptr<float>()
	));
	// cout << out << endl;
	// cout << "run done" << endl;
}

void conv2d_weight(
	torch::Tensor grad_weight, // out
	torch::Tensor input, // in
	torch::Tensor grad_out, // in
	int stride,
	int padding,
	int device_id
) {
	auto cm = CudaSetDeviceManager(); 
	cm.push(device_id); {
		auto sw = grad_weight.sizes();
		auto sin = input.sizes();
		auto sout = grad_out.sizes();
		// auto handle = HCudnnHandle(true);
		cudnnHandle_t handle = at::native::getCudnnHandle();

		// cout << "desc" << endl;
		HCudnnTensorDesc input_desc({(int)sin[0], (int)sin[1], (int)sin[2], (int)sin[3]}, CUDNN_DATA_FLOAT);
		HCudnnTensorDesc out_desc({(int)sout[0], (int)sout[1], (int)sout[2], (int)sout[3]}, CUDNN_DATA_FLOAT);
		HCudnnConvDesc conv_desc({padding,padding}, {stride,stride}, {1,1}, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH);
		HCudnnFilterDesc filter_desc({(int)sout[1], (int)sin[1], 3, 3}, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW);
		
		float one = 1;
		float zero = 0;
		
		cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

		size_t size;
		cuda(cudnnGetConvolutionBackwardFilterWorkspaceSize(
			handle,
			input_desc.data(),
			out_desc.data(),
			conv_desc.data(),
			filter_desc.data(),
			algo,
			&size
		));

		// cout << "workspace" << endl;
		HCudaMemory<uint8_t> workspace(size);
		// cout << "size: " << size/1024/1024 << " MB" << endl;
		

		// cout << "run" << endl;
		cuda(cudnnConvolutionBackwardFilter(
			handle,
			&one,
			input_desc.data(),
			input.data_ptr<float>(),
			out_desc.data(),
			grad_out.data_ptr<float>(),
			conv_desc.data(),
			algo,
			// workspace.data(),
			// 1024*1024*1024,
			nullptr,
			0,
			&zero,
			filter_desc.data(),
			grad_weight.data_ptr<float>()
		));
		// cout << grad_weight << endl;
		// cout << "run done" << endl;
	} cm.pop();
}


void test1() {
	dlopen("/home/hosan/anaconda3/envs/fastmri/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so", RTLD_LAZY);
	auto handle = HCudnnHandle(true);

	cout << "input_desc" << endl;
	torch::Tensor input = torch::ones({1,1,10,10}).to(torch::Device(torch::DeviceType::CUDA, 0));
	
	HCudnnTensorDesc input_desc({1,1,10,10}, CUDNN_DATA_FLOAT);
	cout << "out_desc" << endl;
	torch::Tensor grad_out = torch::ones({1,1,10,10}).to(torch::Device(torch::DeviceType::CUDA, 0));
	HCudnnTensorDesc out_desc({1,1,10,10,}, CUDNN_DATA_FLOAT);
	cout << "conv_desc" << endl;
	torch::Tensor grad_weight = torch::ones({1,1,3,3}).to(torch::Device(torch::DeviceType::CUDA, 0));
	HCudnnConvDesc conv_desc({1,1}, {1,1}, {1,1}, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH);
	float one = 1;
	float zero = 0;
	
	cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
	
	cout << "workspace" << endl;
	HCudaMemory<float> workspace(256*1024*1024);
	cout << "filter_desc" << endl;
	HCudnnFilterDesc filter_desc({1,1,3,3}, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW);

	cout << "run" << endl;
	cuda(cudnnConvolutionBackwardFilter(
		handle.data(),
		&one,
		input_desc.data(),
		input.data_ptr<float>(),
		out_desc.data(),
		grad_out.data_ptr<float>(),
		conv_desc.data(),
		algo,
		workspace.data(),
		1024*1024*1024,
		&zero,
		filter_desc.data(),
		grad_weight.data_ptr<float>()
	));
	cout << "end" << endl;
	cout << grad_weight << endl;
}

void test2() {
	torch::Tensor out = torch::ones({1,1,10,10}).to(torch::Device(torch::DeviceType::CUDA, 0));
	torch::Tensor input = torch::ones({1,1,10,10}).to(torch::Device(torch::DeviceType::CUDA, 0));
	torch::Tensor weight = torch::ones({1,1,3,3}).to(torch::Device(torch::DeviceType::CUDA, 0));
	auto sout = out.sizes();
	auto sin = input.sizes();
	auto sw = weight.sizes();
	
	auto handle = HCudnnHandle(true);
	float one = 1;
	float zero = 0;
	HCudnnTensorDesc out_desc({(int)sout[0], (int)sout[1], (int)sout[2], (int)sout[3]}, CUDNN_DATA_FLOAT);
	HCudnnTensorDesc input_desc({(int)sin[0], (int)sin[1], (int)sin[2], (int)sin[3]}, CUDNN_DATA_FLOAT);
	HCudnnFilterDesc weight_desc({(int)sw[0], (int)sw[1], (int)sw[2], (int)sw[3]}, CUDNN_DATA_FLOAT);
	HCudnnConvDesc conv_desc({1,1},{1,1},{1,1});

	cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

	HCudaMemory<uint8_t> workspace(1024*1024*1024);

	cuda(cudnnConvolutionForward(
		handle.data(), 
		&one, input_desc.data(), input.data_ptr<float>(),
		weight_desc.data(), weight.data_ptr<float>(),
		conv_desc.data(), algo, workspace.data(), workspace.size(),
		&zero, out_desc.data(), out.data_ptr<float>()
	));

	cout << "end" << endl;
	cout << out << endl;
}

void test3() {
	// int version = 0;
	// cudaDriverGetVersion(&version);
	// cout << "cuda driver version: " << version << endl;
	// cudaRuntimeGetVersion(&version);
	// cout << "cuda runtime version: " << version << endl;
	// cout << "cudnn version: " << cudnnGetVersion() << endl;

	// at::native::getCudnnHandle;
	// namespace F = torch::nn::functional;
	// F::conv2d
	// cout << c10::cuda::getCurrentCUDAStream() << endl;
}

int main(char** argc, int argv) {
	dlopen("/home/hosan/anaconda3/envs/fastmri/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so", RTLD_LAZY);
#if 0
	dlopen("/home/hosan/anaconda3/envs/fastmri/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so", RTLD_LAZY);

	torch::Tensor input = torch::ones({1,1,10,10}).to(torch::Device(torch::DeviceType::CUDA, 0));
	cout << input << endl;
#endif
	test3();
	test2();
}

PYBIND11_MODULE(bind, m) {
	m.def("f", &f);
	m.def("conv2d", &conv2d);
	m.def("conv2d_weight", &conv2d_weight);
}