#pragma once
#include "memory_handle.h"
#include "cudnn_handle.h"
#include "cuda_global_setting.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @class	Tensor4d
///////////////////////////////////////////////////////////////////////////////////////////////////
class Tensor4d {
protected:
	HCudaMemory<float> mem_;
	HCudnnTensorDesc desc_;
public:
	void init();
	void constr(const HCudaMemory<float>& mem, const HCudnnTensorDesc& desc);
	Tensor4d();
	Tensor4d(const HCudaMemory<float>& mem, const HCudnnTensorDesc& desc);
	bool valid() const;
	const HCudaMemory<float>& mem() const;
	const HCudnnTensorDesc& desc() const;
	void resize(const HCudnnTensorDesc& desc);
	void fromCPU(const std::vector<float>& c_data);
	void fromCPU(const std::vector<float>& c_data, int begin, int end);
	std::vector<float> toCPU() const;
	std::vector<float> toCPU(int begin, int end) const;
	void toCPU(std::vector<float>& c_data) const;
	friend std::ostream& operator<< (std::ostream& o, const Tensor4d& t);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @class	Weight4d
///////////////////////////////////////////////////////////////////////////////////////////////////
class Weight4d {
protected:
	HCudaMemory<float> mem_;
	HCudnnFilterDesc desc_;
public:
	void init();
	void constr(const HCudaMemory<float>& mem, const HCudnnFilterDesc& desc);
	Weight4d();
	Weight4d(const HCudaMemory<float>& mem, const HCudnnFilterDesc& desc);
	bool valid() const;
	const HCudaMemory<float>& mem() const;
	const HCudnnFilterDesc& desc() const;
	void fromCPU(const std::vector<float>& c_data);
	std::vector<float> toCPU() const;
	friend std::ostream& operator<< (std::ostream& o, const Weight4d& t);
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @class	Bias4d
///////////////////////////////////////////////////////////////////////////////////////////////////
class Bias4d : public Tensor4d {
public:
	void init();
	void constr(const HCudaMemory<float>& mem, const HCudnnTensorDesc& desc);
	Bias4d();
	Bias4d(const HCudaMemory<float>& mem, const HCudnnTensorDesc& desc);
};
