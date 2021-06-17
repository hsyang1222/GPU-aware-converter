#include "cudnn_handle.h"
#include "cout_overloading.h"
#include <cublas_v2.h>
#include <iostream>

#define DEBUG_MEMORY 0
#define DEBUG_WARNING 0
#define DEBUG_DIM 0
#define CUDNN_DATA_REAL CUDNN_DATA_FLOAT
#define REALLOC_MEMORY 0

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCublasHandle
///////////////////////////////////////////////////////////////////////////////////////////////////

void HCublasHandle::init() {
	data_ = nullptr;
}
HCublasHandle::HCublasHandle() {
	init();
}
HCublasHandle::HCublasHandle(bool alloc) {
	init();
	if (alloc) {
		cublasHandle_t* raw = new cublasHandle_t();
		cuda(cublasCreate(raw));
#if DEBUG_MEMORY
		std::cout << "HCublasHandle::gen: raw(" << *raw << ")" << std::endl;
#endif
		data_ = std::shared_ptr<cublasHandle_t>(raw, [](cublasHandle_t* raw) {
#if DEBUG_MEMORY
			std::cout << "HCublasHandle::deleter: raw(" << *raw << ")" << std::endl;
#endif
			cuda(cublasDestroy(*raw));
			delete raw;
		});
	}
}
bool HCublasHandle::valid() const {
	return data_ != nullptr;
}
cublasHandle_t HCublasHandle::data() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCublasHandle::data: !valid()");
	}
#endif
	return *data_;
}
std::ostream& operator << (std::ostream& o, const HCublasHandle& h) {
	if (h.valid()) {
		o << "HCublasHandle[valid=" << h.valid() << ", data=" << h.data() << "]";
	}
	else {
		o << "HCublasHandle[valid=" << h.valid() << "]";
	}
	return o;
}



///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudnnHandle
///////////////////////////////////////////////////////////////////////////////////////////////////
void HCudnnHandle::init() {
	data_ = nullptr;
}
HCudnnHandle::HCudnnHandle() {
	init();
}
HCudnnHandle::HCudnnHandle(bool alloc) {
	init();
	if (alloc) {
		cudnnHandle_t* raw = new cudnnHandle_t();
		cuda(cudnnCreate(raw));
#if DEBUG_MEMORY
		std::cout << "HCudnnHandle::gen: raw(" << *raw << ")" << std::endl;
#endif
		data_ = std::shared_ptr<cudnnHandle_t>(raw, [](cudnnHandle_t* raw) {
#if DEBUG_MEMORY
			std::cout << "HCudnnHandle::deleter: raw(" << *raw << ")" << std::endl;
#endif
			cuda(cudnnDestroy(*raw));
			delete raw;
		});
	}
}
bool HCudnnHandle::valid() const {
	return data_ != nullptr;
}
cudnnHandle_t HCudnnHandle::data() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnHandle::data: !valid()");
	}
#endif
	return *data_;
}
std::ostream& operator << (std::ostream& o, const HCudnnHandle& h) {
	if (h.valid()) {
		o << "HCudnnHandle[valid=" << h.valid() << ", data=" << h.data() << "]";
	}
	else {
		o << "HCudnnHandle[valid=" << h.valid() << "]";
	}
	return o;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudnnActivationDesc
///////////////////////////////////////////////////////////////////////////////////////////////////
void HCudnnActivationDesc::init() {
	data_ = nullptr;
}
HCudnnActivationDesc::HCudnnActivationDesc() {
	init();
}
HCudnnActivationDesc::HCudnnActivationDesc(bool alloc) {
	init();
	if (alloc) {
		cudnnActivationDescriptor_t* raw = new cudnnActivationDescriptor_t();
		cuda(cudnnCreateActivationDescriptor(raw));
		cuda(cudnnSetActivationDescriptor(*raw, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
#if DEBUG_MEMORY
		std::cout << "HCudnnActivationDesc::gen: raw(" << *raw << ")" << std::endl;
#endif
		data_ = std::shared_ptr<cudnnActivationDescriptor_t>(raw, [](cudnnActivationDescriptor_t* raw) {
#if DEBUG_MEMORY
			std::cout << "HCudnnActivationDesc::deleter: raw(" << *raw << ")" << std::endl;
#endif
			cuda(cudnnDestroyActivationDescriptor(*raw));
			delete raw;
		});
	}
}
bool HCudnnActivationDesc::valid() const {
	return data_ != nullptr;
}
cudnnActivationDescriptor_t HCudnnActivationDesc::data() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnActivationDesc::data: !valid()");
	}
#endif
	return *data_;
}
std::ostream& operator << (std::ostream& o, const HCudnnActivationDesc& h) {
	if (h.valid()) {
		o << "HCudnnActivationDesc[valid=" << h.valid() << ", data=" << h.data() << "]";
	}
	else {
		o << "HCudnnActivationDesc[valid=" << h.valid() << "]";
	}
	return o;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudnnTensorDesc
///////////////////////////////////////////////////////////////////////////////////////////////////
void HCudnnTensorDesc::init() {
	data_ = nullptr;
	dims_.clear();
	cstrides_.clear();
}
void HCudnnTensorDesc::constr(const std::vector<int>& dims, const std::vector<int>& cstrides, cudnnDataType_t dtype) {
#if DEBUG_DIM
	if (dims.size() != cstrides.size()) {
		_raise("HCudnnTensorDesc::constr: dims.size != cstrides.size");
	}
#endif
	dims_ = dims;
	cstrides_ = cstrides;
	cudnnTensorDescriptor_t* raw = new cudnnTensorDescriptor_t();
	cuda(cudnnCreateTensorDescriptor(raw));
	cuda(cudnnSetTensorNdDescriptor(*raw, dtype, dims_.size(), dims_.data(), cstrides_.data()));
#if DEBUG_MEMORY
	std::cout << "HCudnnTensorDesc::constr: raw(" << *raw << ")" << std::endl;
#endif
	data_ = std::shared_ptr<cudnnTensorDescriptor_t>(raw, [](cudnnTensorDescriptor_t* raw) {
#if DEBUG_MEMORY
		std::cout << "HCudnnTensorDesc::deleter: raw(" << *raw << ")" << std::endl;
#endif 
		cuda(cudnnDestroyTensorDescriptor(*raw));
		delete raw;
	});
}
HCudnnTensorDesc::HCudnnTensorDesc() {
	init();
}
HCudnnTensorDesc::HCudnnTensorDesc(const std::vector<int>& dims, cudnnDataType_t dtype, cudnnTensorFormat_t format) {
	init();
	constr(dims, VectorHelper::gen_cstrides(dims, format), dtype);
}
//HCudnnTensorDesc::HCudnnTensorDesc(const std::vector<int> dims, const std::vector<int>& cstrides) {
//	init();
//	constr(dims, cstrides);
//}
bool HCudnnTensorDesc::valid() const {
#if DEBUG_DIM
	if (dims_.size() != cstrides_.size()) {
		_raise("HCudnnTensorDesc::valid: dims_.size != cstrides_.size");
	}
#endif
	return data_ && dims_.size() > 0 && cstrides_.size() > 0;
}
cudnnTensorDescriptor_t HCudnnTensorDesc::data() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnTensorDesc::data: !valid()");
	}
#endif
	return *data_;
}
const std::vector<int>& HCudnnTensorDesc::dims() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnTensorDesc::dims: !valid()");
	}
#endif
	return dims_;
}
const std::vector<int>& HCudnnTensorDesc::cstrides() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnTensorDesc::cstrides: !valid()");
	}
#endif
	return cstrides_;
}
size_t HCudnnTensorDesc::sizeByDims() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnTensorDesc::sizeByDims: !valid()");
	}
#endif
	return VectorHelper::prod(dims_);
}
std::ostream& operator << (std::ostream& o, const HCudnnTensorDesc& h) {
	if (h.valid()) {
		o << "HCudnnTensorDesc[valid=" << h.valid() << ", dims=" << h.dims() << ", cstrides=" << h.cstrides() << ", data=" << h.data() << "]";
	}
	else {
		o << "HCudnnTensorDesc[valid=" << h.valid() << "]";
	}
	return o;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudnnFilterDesc
///////////////////////////////////////////////////////////////////////////////////////////////////
void HCudnnFilterDesc::init() {
	data_ = nullptr;
	dims_.clear();
}
HCudnnFilterDesc::HCudnnFilterDesc() {
	init();
}
HCudnnFilterDesc::HCudnnFilterDesc(const std::vector<int>& dims, cudnnDataType_t dtype, cudnnTensorFormat_t format) {
	init();
	dims_ = dims;
	cudnnFilterDescriptor_t* raw = new cudnnFilterDescriptor_t();
	cuda(cudnnCreateFilterDescriptor(raw));
	cuda(cudnnSetFilterNdDescriptor(*raw, dtype, format, dims_.size(), dims_.data()));
#if DEBUG_MEMORY
	//std::cout << "HCudnnFilterDesc::constr: raw(" << *raw << ")" << std::endl;
#endif
	data_ = std::shared_ptr<cudnnFilterDescriptor_t>(raw, [](cudnnFilterDescriptor_t* raw) {
#if DEBUG_MEMORY
		std::cout << "HCudnnFilterDesc::deleter: raw(" << *raw << ")" << std::endl;
#endif 
		cuda(cudnnDestroyFilterDescriptor(*raw));
		delete raw;
	});

}
bool HCudnnFilterDesc::valid() const {
	return data_ && dims_.size() > 0;
}
cudnnFilterDescriptor_t HCudnnFilterDesc::data() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnFilterDesc::data: !valid()");
	}
#endif
	return *data_;
}
const std::vector<int>& HCudnnFilterDesc::dims() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnFilterDesc::dims: !valid()");
	}
#endif
	return dims_;
}
size_t HCudnnFilterDesc::sizeByDims() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnFilterDesc::sizeByDims: !valid()");
	}
#endif
	return VectorHelper::prod(dims_);
}
std::ostream& operator<< (std::ostream& o, const HCudnnFilterDesc& h) {
	if (h.valid()) {
		o << "HCudnnFilterDesc[valid=" << h.valid() << ", dims=" << h.dims() << ", data=" << h.data() << "]";
	}
	else {
		o << "HCudnnFilterDesc[valid=" << h.valid() << "]";
	}
	return o;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudnnConvDesc
///////////////////////////////////////////////////////////////////////////////////////////////////
void HCudnnConvDesc::init() {
	data_ = nullptr;
	pads_.clear();
	strides_.clear();
	dilations_.clear();
}
void HCudnnConvDesc::constr(const std::vector<int>& pads, const std::vector<int>& strides, const std::vector<int>& dilations, cudnnDataType_t dtype, cudnnMathType_t mtype) {
#if DEBUG_DIM
	if (pads.size() != strides.size() || pads.size() != dilations.size()) {
		_raise("HCudnnConvDesc::constr: pads.size, strides.size, dilations.size not same");
	}
#endif
	pads_ = pads;
	strides_ = strides;
	dilations_ = dilations;
	cudnnConvolutionDescriptor_t* raw = new cudnnConvolutionDescriptor_t();
	cuda(cudnnCreateConvolutionDescriptor(raw));
	cuda(cudnnSetConvolutionNdDescriptor(*raw, pads.size(), pads.data(), strides.data(), dilations.data(), CUDNN_CROSS_CORRELATION, dtype));
	cuda(cudnnSetConvolutionMathType(*raw, mtype))
#if DEBUG_MEMORY
	std::cout << "HCudnnConvDesc::constr: raw(" << *raw << ")" << std::endl;
#endif
	data_ = std::shared_ptr<cudnnConvolutionDescriptor_t>(raw, [](cudnnConvolutionDescriptor_t* raw) {
#if DEBUG_MEMORY
		std::cout << "HCudnnConvDesc::deleter: raw(" << *raw << ")" << std::endl;
#endif
		cuda(cudnnDestroyConvolutionDescriptor(*raw));
		delete raw;
	});
}
HCudnnConvDesc::HCudnnConvDesc() {
	init();
}
HCudnnConvDesc::HCudnnConvDesc(const std::vector<int>& pads, cudnnDataType_t dtype, cudnnMathType_t mtype) {
	init();
	constr(pads, std::vector<int>(pads.size(), 1), std::vector<int>(pads.size(), 1), dtype, mtype);
}
HCudnnConvDesc::HCudnnConvDesc(const std::vector<int>& pads, const std::vector<int>& strides, cudnnDataType_t dtype, cudnnMathType_t mtype) {
	init();
	constr(pads, strides, std::vector<int>(pads.size(), 1), dtype, mtype);
}
HCudnnConvDesc::HCudnnConvDesc(const std::vector<int>& pads, const std::vector<int>& strides, const std::vector<int>& dilations, cudnnDataType_t dtype, cudnnMathType_t mtype) {
	init();
	constr(pads, strides, dilations, dtype, mtype);
}
bool HCudnnConvDesc::valid() const {
#if DEBUG_DIM
	if (pads_.size() != strides_.size() || pads_.size() != dilations_.size()) {
		_raise("HCudnnConvDesc::constr: pads_.size, strides_.size, dilations_.size not same");
	}
#endif
	return data_ && pads_.size() > 0 && strides_.size() > 0 && dilations_.size() > 0;
}
cudnnConvolutionDescriptor_t HCudnnConvDesc::data() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnConvDesc::data: !valid()");
	}
#endif
	return *data_;
}
const std::vector<int>& HCudnnConvDesc::pads() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnConvDesc::pads: !valid()");
	}
#endif
	return pads_;
}
const std::vector<int>& HCudnnConvDesc::strides() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnConvDesc::strides: !valid()");
	}
#endif
	return strides_;
}
const std::vector<int>& HCudnnConvDesc::dilatinos() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudnnConvDesc::dilatinos: !valid()");
	}
#endif
	return dilations_;
}
std::ostream& operator<< (std::ostream& o, const HCudnnConvDesc& h) {
	if (h.valid()) {
		o << "HCudnnConvDesc[valid=" << h.valid() << ", pads=" << h.pads() << ", strides=" << h.strides() << ", dilations=" << h.dilatinos() << ", data=" << h.data() << "]";
	}
	else {

	}
	return o;
}



///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudaByteMemory
///////////////////////////////////////////////////////////////////////////////////////////////////
#if 0
void HCudaByteMemory::init() {
	data_ = nullptr;
	sizeInBytes_ = 0;
}
HCudaByteMemory::HCudaByteMemory() {
	init();
}
HCudaByteMemory::HCudaByteMemory(size_t sizeInBytes) {
	init();
	sizeInBytes_ = sizeInBytes;
	void* raw;
	cuda(cudaMalloc(&raw, sizeInBytes));
#if DEBUG_MEMORY
	std::cout << "HCudaByteMemory::HCudaByteMemory: raw(" << raw << "), sizeInBytes(" << sizeInBytes << ")" << std::endl;
#endif
	data_ = std::shared_ptr<void>(raw, [](void* raw) {
#if DEBUG_MEMORY
		if (!raw) {
			try {
				// https://akrzemi1.wordpress.com/2011/09/21/destructors-that-throw/
				// https://stackoverflow.com/questions/13778664/how-can-shared-ptr-offer-noexcept-assignment
				// https://stackoverflow.com/questions/19528691/why-do-the-std-smart-pointer-type-destructors-not-inherit-the-noexcept-dtor-stat
				_raise("HCudaByteMemory::deleter: raw is nullptr");
			}
			catch (std::exception& e) {
				std::cout << "HCudaByteMemory::deleter: ! " << e << std::endl;
			}
		}
		std::cout << "HCudaByteMemory::deleter: delete(" << raw << ")" << std::endl;
#endif
		cuda(cudaFree(raw));
	});
}
bool HCudaByteMemory::valid() const {
	return data_ != nullptr;
}
void* HCudaByteMemory::data() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaByteMemory::data: !valid()");
	}
#endif
	return data_.get();
}
size_t HCudaByteMemory::sizeInBytes() const {
	return sizeInBytes_;
} ///< memory size in byte unit.
std::ostream& operator<< (std::ostream& o, const HCudaByteMemory& h) {
	if (h.valid()) {
		o << "HCudaByteMemory[valid=" << h.valid() << ", sizeInBytes=" << h.sizeInBytes() << ", data=" << h.data() << "]";
	}
	else {
		o << "HCudaByteMemory[valid=" << h.valid() << ", sizeInBytes=" << h.sizeInBytes() << "]";
	}
	return o;
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @VectorHelper
///////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<int> VectorHelper::gen_cstrides(const std::vector<int>& dims, cudnnTensorFormat_t format) {
	std::vector<int> res(dims.size(), 1);
	if (format == CUDNN_TENSOR_NCHW) {
		for (int i = 0; i < res.size() - 1; i++) {
			for (int j = i + 1; j < res.size(); j++) {
				res[i] *= dims[j];
			}
		}
	}
	else { // CUDNN_TENSOR_NHWC
		//for (int i = 2; i < res.size(); i++) {
		//	for (int j = i ; j < res.size(); j++) {
		//		res[i] *= dims[j];
		//	}
		//}
		//res[0] = dims[2] * res[2];
		
		res[1] = 1;
		res[res.size() - 1] = dims[1];
		for (int i = res.size() - 2; i >= 2; i--) {
			res[i] = res[i + 1] * dims[i];
		}
		res[0] = res[2] * dims[2];
	}
	return res;
}
size_t VectorHelper::prod(const std::vector<int>& dims) {
	size_t res = 1;
	for (auto e : dims) {
		res *= e;
	}
	return res;
}
size_t VectorHelper::idx(const std::vector<int>& i, const std::vector<int>& cstrides) {
	size_t res = 0;
	for (int j = 0; j < i.size(); j++) {
		res += cstrides[j] * i[j];
	}
	return res;
}