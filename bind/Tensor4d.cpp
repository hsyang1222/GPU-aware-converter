#include "Tensor4d.h"
using namespace std;

#define DEBUG_MEMORY 0
#define DEBUG_WARNING 0
#define DEBUG_DIM 0
#define CUDNN_DATA_REAL CUDNN_DATA_FLOAT
#define REALLOC_MEMORY 0

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @class	Tensor4d
///////////////////////////////////////////////////////////////////////////////////////////////////

void Tensor4d::init() {
	mem_.init();
	desc_.init();
}

void Tensor4d::constr(const HCudaMemory<float>& mem, const HCudnnTensorDesc& desc) {
	init();
#if DEBUG_MEMORY
	if (!mem.valid() || !desc.valid()) {
		_raise("Tensor4d::constr: not valid");
	}
	if (mem.size() < desc.sizeByDims()) {
		stringstream ss;
		ss << "Tensor4d::constr: mem.size(" << mem.size() << ") < desc.numel(" << desc.sizeByDims() << ")" << endl;
		ss << "desc.dims=" << desc.dims() << endl;
		_raise(ss.str().c_str());
	}
#endif
	mem_ = mem;
	desc_ = desc;
}

Tensor4d::Tensor4d() {
	init();
}

Tensor4d::Tensor4d(const HCudaMemory<float>& mem, const HCudnnTensorDesc& desc) {
	constr(mem, desc);
}

bool Tensor4d::valid() const {
	return mem_.valid() && desc_.valid();
}

const HCudaMemory<float>& Tensor4d::mem() const {
	if (!mem_.valid()) {
		_raise("Tensor4d::mem: not valid");
	}
	return mem_;
}

const HCudnnTensorDesc& Tensor4d::desc() const {
	if (!desc_.valid()) {
		_raise("Tensor4::mem: not valid");
	}
	return desc_;
}

void Tensor4d::resize(const HCudnnTensorDesc& desc) {
	if (desc.dims() != desc_.dims()) {
		if (desc.sizeByDims() > mem_.size()) {
#if DEBUG_MEMORY
			lg << "Tensor4d::resize: realloc" << endl;
#endif
			mem_ = HCudaMemory<float>(desc.sizeByDims());
			desc_ = desc;
		}
		else if (desc.sizeByDims() <= mem_.size()) {
			desc_ = desc;
		}
	}
}

void Tensor4d::fromCPU(const std::vector<float>& c_data) {
	mem_.fromCPU(c_data);
}

void Tensor4d::fromCPU(const std::vector<float>& c_data, int begin, int end) {
	mem_.fromCPU(c_data, begin, end);
}

std::vector<float> Tensor4d::toCPU() const {
	return mem_.toCPU();
}

std::vector<float> Tensor4d::toCPU(int begin, int end) const {
	vector<float> res(end - begin);
	mem_.toCPU(res, begin, end);
	return res;
}

void Tensor4d::toCPU(std::vector<float>& c_data) const {
	mem_.toCPU(c_data);
}

std::ostream& operator<< (std::ostream& o, const Tensor4d& t) {
	o << "Tensor4d { valid: " << t.valid() << ", mem: " << t.mem() << ", desc: " << t.desc(), "}";
	return o;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
/// @class	Weight4d
///////////////////////////////////////////////////////////////////////////////////////////////////

void Weight4d::init() {
	mem_.init();
	desc_.init();
}

void Weight4d::constr(const HCudaMemory<float>& mem, const HCudnnFilterDesc& desc) {
	init();
#if DEBUG_MEMORY
	if (!mem.valid() || !desc.valid()) {
		_raise("Weight4d::constr: not valid");
	}
	if (mem.size() < desc.sizeByDims()) {
		_raise("Weight4d::constr: mem.size < desc.numel");
	}
#endif
	mem_ = mem;
	desc_ = desc;
}

Weight4d::Weight4d() {
	init();
}

Weight4d::Weight4d(const HCudaMemory<float>& mem, const HCudnnFilterDesc& desc) {
	constr(mem, desc);
}

bool Weight4d::valid() const {
	return mem_.valid() && desc_.valid();
}

const HCudaMemory<float>&  Weight4d::mem() const {
	return mem_;
}

const HCudnnFilterDesc& Weight4d::desc() const {
	return desc_;
}

void Weight4d::fromCPU(const std::vector<float>& c_data) {
	mem_.fromCPU(c_data);
}

std::vector<float> Weight4d::toCPU() const {
	return mem_.toCPU();
}

std::ostream& operator<< (std::ostream& o, const Weight4d& t) {
	o << "Weight4d { valid: " << t.valid() << ", mem: " << t.mem() << ", desc: " << t.desc() << "}";
	return o;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
/// @class	Bias4d
///////////////////////////////////////////////////////////////////////////////////////////////////

void Bias4d::init() {
	Tensor4d::init();
}

void Bias4d::constr(const HCudaMemory<float>& mem, const HCudnnTensorDesc& desc) {
	init();
	Tensor4d::constr(mem, desc);
}

Bias4d::Bias4d() : Tensor4d() {
	init();
}

Bias4d::Bias4d(const HCudaMemory<float>& mem, const HCudnnTensorDesc& desc) {
	constr(mem, desc);
}