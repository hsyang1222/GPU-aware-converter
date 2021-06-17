#pragma once
#include "cudnn_handle.h"
#include "Tensor4d.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @class	Layer4d
/// 		singleton pattern
///////////////////////////////////////////////////////////////////////////////////////////////////
class Layer4d {
private:
	HCudnnHandle cudnn_;
	HCublasHandle cublas_;
	HCudnnActivationDesc desc_relu_;
	HCudaMemory<uint8_t> mem_workspace_;
public:
	Layer4d();
	Layer4d(bool alloc);
	void init();
	void constr(bool alloc);
	const HCudnnHandle& getCudnn() const;
	const HCublasHandle& getCublas() const;
	void conv2d(
		Tensor4d& out,
		const Tensor4d& in,
		const Weight4d& weight,
		const Bias4d& bias,
		const std::vector<int>& strides = { 1,1 },
		const std::vector<int>& padding = { 1,1 }
	);
	void add(
		Tensor4d& out,
		const Tensor4d& in1
	) const;
	void add(
		Tensor4d& out,
		const Tensor4d& in1,
		const Tensor4d& in2
	) const;
	void copy(
		Tensor4d& out,
		const Tensor4d& in
	) const;
};


