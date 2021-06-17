#pragma once
#include <memory>
#include <vector>
#include "cuda_global_setting.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
///	@HCudaMemory
///////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class HCudaMemory {
public:
	HCudaMemory();
	HCudaMemory(size_t size);
	HCudaMemory(T* data, size_t size, bool is_released);

public:
	void init();
	bool valid() const;
	T* data() const;
	size_t size() const;
	void setZero();
	void fromCPU(const std::vector<T>& c_data);
	void fromCPU(const std::vector<T>& c_data, int begin, int end);
	void fromGPU(T* data, size_t size, size_t offset);
	std::vector<T> toCPU() const;
	void toCPU(std::vector<T>& c_data) const;
	void toCPU(std::vector<T>& c_data, int begin, int end) const;
	template<class T2>
	friend std::ostream& operator<< (std::ostream& o, const HCudaMemory<T2>& h);

private:
	std::shared_ptr<T> data_ = nullptr;
	size_t	size_ = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudaMemory2D
///////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class HCudaMemory2D
{
public:
	HCudaMemory2D();
	HCudaMemory2D(size_t nx, size_t ny, size_t pitch);
	HCudaMemory2D(size_t nx, size_t ny);
	HCudaMemory2D(HCudaMemory2D<T> &data, bool is_released);
	HCudaMemory2D(size_t nx, size_t ny, size_t pitch, T* data, bool is_released);

public:
	void init();
	bool valid() const;
	T* data() const;
	size_t nx() const;
	size_t ny() const;
	size_t pitch() const;
	void setZero();
	void fromCPU(const std::vector<T>& c_data);
	void fromCPU(const std::vector<T>& c_data, int begin, int end);
	void fromGPU1D(T* data, size_t size, size_t offset);
	void fromGPU2D(HCudaMemory2D<T> &data);
	void toGPU1D(T* data, size_t offset);
	std::vector<T> toCPU() const;
	void toCPU(std::vector<T>& c_data) const;
	void toCPU(std::vector<T>& c_data, int begin, int end) const;
	template<class T2>
	friend std::ostream& operator<< (std::ostream& o, const HCudaMemory2D<T2>& h);

private:
	std::shared_ptr<T> data_ = nullptr;
	size_t	nx_ = 0;
	size_t	pitch_ = 0;
	size_t	ny_ = 0;
};


///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudArray2DMemory
///////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class HCudaArray2DMemory
{
public:
	HCudaArray2DMemory();
	HCudaArray2DMemory(size_t nx, size_t ny);

public:
	void init();
	bool valid() const;
	cudaArray* data() const;
	size_t nx() const;
	size_t ny() const;
	void fromCPU(const std::vector<T>& c_data);
	void fromGPU(T* g_data);
	template<class T2>
	friend std::ostream& operator<< (std::ostream& o, const HCudaArray2DMemory<T2>& h);

private:
	std::shared_ptr<cudaArray> data_ = nullptr;
	size_t nx_ = 0;
	size_t ny_ = 0;
	cudaChannelFormatDesc channel_desc_ = { 0 };
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudaArray3DMemory
///////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class HCudaArray3DMemory
{
public:
	HCudaArray3DMemory();
	HCudaArray3DMemory(size_t nx, size_t ny, size_t nz, bool isLayered = false);

public:
	void init();
	bool valid() const;
	cudaArray* data() const;
	cudaExtent size() const;
	void fromCPU(const std::vector<T>& c_data);
	void fromGPU(T* g_data);
	cudaMemcpy3DParms* get_copy_params();
	template<class T2>
	friend std::ostream& operator<< (std::ostream& o, const HCudaArray3DMemory<T2>& h);

private:
	std::shared_ptr<cudaArray> data_ = nullptr;
	cudaExtent size_;
	cudaChannelFormatDesc channel_desc_ = { 0 };
	cudaMemcpy3DParms copy_params_ = { 0 };
};

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudaTextureObject
///////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
class HCudaTextureObject
{
public:
	HCudaTextureObject();
	HCudaTextureObject(HCudaMemory2D<T> &mem, cudaTextureReadMode read_mode, cudaTextureFilterMode filter_mode, std::vector<cudaTextureAddressMode> address_mode);
	HCudaTextureObject(size_t nx, size_t ny, cudaTextureReadMode read_mode, cudaTextureFilterMode filter_mode, std::vector<cudaTextureAddressMode> address_mode);
	HCudaTextureObject(HCudaArray2DMemory<T> arr, cudaTextureReadMode read_mode, cudaTextureFilterMode filter_mode, std::vector<cudaTextureAddressMode> address_mode);
	HCudaTextureObject(size_t nx, size_t ny, size_t nz, cudaTextureReadMode read_mode, cudaTextureFilterMode filter_mode, std::vector<cudaTextureAddressMode> address_mode, bool isLayered = false);
	HCudaTextureObject(HCudaArray3DMemory<T> arr, cudaTextureReadMode read_mode, cudaTextureFilterMode filter_mode, std::vector<cudaTextureAddressMode> address_mode);

public:
	void init();
	bool valid() const;
	cudaTextureObject_t data() const;
	void fromCPUto3D(const std::vector<T>& c_data);
	void fromGPUto3D(T* g_data);
	void fromCPUto2D(const std::vector<T>& c_data);
	void fromGPUto2D(T* g_data);
	void fromCPUto2Dmem(const std::vector<T>& c_data);
	void fromGPUto2Dmem(T* g_data);
	HCudaMemory2D<T> mem2D() const;


	template<class T2>
	friend std::ostream& operator<< (std::ostream& o, const HCudaTextureObject<T2>& h);
private:
	HCudaArray2DMemory<T> array2D_;
	HCudaArray3DMemory<T> array3D_;
	HCudaMemory2D<T>	mem2D_;
	std::shared_ptr<cudaTextureObject_t> data_;
	cudaResourceDesc res_desc_;
	cudaTextureDesc tex_desc_;
};

#include "memory_handle_template_impl.h"
