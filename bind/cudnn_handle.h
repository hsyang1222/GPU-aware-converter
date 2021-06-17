#pragma once
#include <memory>
#include <vector>
#include "cuda_global_setting.h"

#define DEBUG_MEMORY 0
#define DEBUG_WARNING 0
#define DEBUG_DIM 0
#define CUDNN_DATA_REAL CUDNN_DATA_FLOAT
#define REALLOC_MEMORY 0

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCublasHandle
///////////////////////////////////////////////////////////////////////////////////////////////////
class HCublasHandle {

public:
	void init();
	HCublasHandle();
	HCublasHandle(bool alloc);
	bool valid() const;
	cublasHandle_t data() const;
	friend std::ostream& operator << (std::ostream& o, const HCublasHandle& h);
private:
	std::shared_ptr<cublasHandle_t> data_;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudnnHandle
///////////////////////////////////////////////////////////////////////////////////////////////////
class HCudnnHandle {
public:
	void init();
	HCudnnHandle();
	HCudnnHandle(bool alloc);
	bool valid() const;
	cudnnHandle_t data() const;
	friend std::ostream& operator << (std::ostream& o, const HCudnnHandle& h);
private:
	std::shared_ptr<cudnnHandle_t> data_;

};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudnnActivationDesc
///////////////////////////////////////////////////////////////////////////////////////////////////
class HCudnnActivationDesc {
public:
	void init();
	HCudnnActivationDesc();
	HCudnnActivationDesc(bool alloc);
	bool valid() const;
	cudnnActivationDescriptor_t data() const;
	friend std::ostream& operator << (std::ostream& o, const HCudnnActivationDesc& h);
private:
	std::shared_ptr<cudnnActivationDescriptor_t> data_;

};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudnnTensorDesc
/// @note	cstrides는 아직 dims 에서 만들어져야함. 메모리 구성등이 오로지 dims를 기준으로 구성되어있음.
///////////////////////////////////////////////////////////////////////////////////////////////////
class HCudnnTensorDesc {
public:
	void init();
	void constr(const std::vector<int>& dims, const std::vector<int>& cstrides, cudnnDataType_t dtype);
	HCudnnTensorDesc();
	HCudnnTensorDesc(const std::vector<int>& dims, cudnnDataType_t dtype=CUDNN_DATA_REAL, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW);
	//HCudnnTensorDesc(const std::vector<int> dims, const std::vector<int>& cstrides);
	bool valid() const;
	cudnnTensorDescriptor_t data() const;
	const std::vector<int>& dims() const;
	const std::vector<int>& cstrides() const;
	size_t sizeByDims() const;
	friend std::ostream& operator << (std::ostream& o, const HCudnnTensorDesc& h);
private:
	std::shared_ptr<cudnnTensorDescriptor_t> data_;
	std::vector<int> dims_;
	std::vector<int> cstrides_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudnnFilterDesc
///////////////////////////////////////////////////////////////////////////////////////////////////
class HCudnnFilterDesc {
public:
	void init();
	HCudnnFilterDesc();
	HCudnnFilterDesc(const std::vector<int>& dims, cudnnDataType_t dtype=CUDNN_DATA_REAL, cudnnTensorFormat_t format=CUDNN_TENSOR_NCHW);
	bool valid() const;
	cudnnFilterDescriptor_t data() const;
	const std::vector<int>& dims() const;
	size_t sizeByDims() const;
	friend std::ostream& operator<< (std::ostream& o, const HCudnnFilterDesc& h);

private:
	std::shared_ptr<cudnnFilterDescriptor_t> data_;
	std::vector<int> dims_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudnnConvDesc
///////////////////////////////////////////////////////////////////////////////////////////////////
class HCudnnConvDesc {
public:
	void init();
	void constr(
		const std::vector<int>& pads, 
		const std::vector<int>& strides, 
		const std::vector<int>& dilations,
		cudnnDataType_t dtype,
		cudnnMathType_t mtype);
	HCudnnConvDesc();
	HCudnnConvDesc(
		const std::vector<int>& pads, 
		cudnnDataType_t dtype = CUDNN_DATA_REAL,
		cudnnMathType_t mtype = CUDNN_TENSOR_OP_MATH);
	HCudnnConvDesc(
		const std::vector<int>& pads,
		const std::vector<int>& strides,
		cudnnDataType_t dtype = CUDNN_DATA_REAL,
		cudnnMathType_t mtype = CUDNN_TENSOR_OP_MATH);
	HCudnnConvDesc(
		const std::vector<int>& pads,
		const std::vector<int>& strides,
		const std::vector<int>& dilations,
		cudnnDataType_t dtype = CUDNN_DATA_REAL,
		cudnnMathType_t mtype = CUDNN_TENSOR_OP_MATH);
	bool valid() const;
	cudnnConvolutionDescriptor_t data() const;
	const std::vector<int>& pads() const;
	const std::vector<int>& strides() const;
	const std::vector<int>& dilatinos() const;
	friend std::ostream& operator<< (std::ostream& o, const HCudnnConvDesc& h);
private:
	std::shared_ptr<cudnnConvolutionDescriptor_t> data_;
	std::vector<int> pads_;
	std::vector<int> strides_;
	std::vector<int> dilations_;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudaByteMemory
///////////////////////////////////////////////////////////////////////////////////////////////////
#if 0
class HCudaByteMemory {
public:
	void init();
	HCudaByteMemory();
	HCudaByteMemory(size_t sizeInBytes);
	bool valid() const;
	void* data() const;
	size_t sizeInBytes() const; ///< memory size in byte unit.
	friend std::ostream& operator<< (std::ostream& o, const HCudaByteMemory& h);
private:
	std::shared_ptr<void> data_;
	size_t sizeInBytes_;

};
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @VectorHelper
///////////////////////////////////////////////////////////////////////////////////////////////////
class VectorHelper {
public:
	static std::vector<int> gen_cstrides(const std::vector<int>& dims, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW);
	static size_t prod(const std::vector<int>& dims);
	static size_t idx(const std::vector<int>& i, const std::vector<int>& cstrides);
};