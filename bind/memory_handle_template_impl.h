#include "memory_handle.h"
#include <iostream>

#define DEBUG_MEMORY 0
#define DEBUG_WARNING 0
#define DEBUG_DIM 0
#define CUDNN_DATA_REAL CUDNN_DATA_FLOAT
#define REALLOC_MEMORY 0

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudaMemory
///////////////////////////////////////////////////////////////////////////////////////////////////

template<class T>
HCudaMemory<T>::HCudaMemory() {

}

template<class T>
HCudaMemory<T>::HCudaMemory(size_t size)
	:size_(size)
{
	T* raw;
	cuda(cudaMalloc(&raw, size_ * sizeof(T)));
#if DEBUG_MEMORY
	std::cout << "HCudaMemory::HCudaMemory: raw(" << uint64_t(raw) << "), size(" << size << ")" << std::endl;
#endif
	data_ = std::shared_ptr<T>(raw, [](T* raw) {
		try {
#if DEBUG_MEMORY
			if (!raw) {
				_raise("HCudaMemory::deleter: raw is nullptr");
			}
			std::cout << "HCudaMemory::deleter: delete(" << uint64_t(raw) << ")" << std::endl;
#endif
			cuda(cudaFree(raw));
		}
		catch (std::exception& e) {
			std::cout << e.what() << std::endl;
		}
	});
}

template<class T>
HCudaMemory<T>::HCudaMemory(T* data, size_t size, bool is_released)
{
	init();
	this->size_ = size;

	data_ = std::shared_ptr<T>(data, [is_released](T* raw) {
		try {
			if (is_released)
			{
#if DEBUG_MEMORY
				if (!raw) {
					_raise("HCudaMemory::deleter: raw is nullptr");
				}
				std::cout << "HCudaMemory::deleter: delete(" << raw << ")" << std::endl;
#endif
				cuda(cudaFree(raw));
			}
		}
		catch (std::exception& e) {
			std::cout << e.what() << std::endl;
		}
	});

}

template<class T>
void HCudaMemory<T>::init()
{
	data_ = nullptr;
	size_ = 0;
}

template<class T>
bool HCudaMemory<T>::valid() const
{
	return data_ != nullptr;
}

template<class T>
T* HCudaMemory<T>::data() const
{
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaMemory::data: !valid()")
	}
#endif

	return data_.get();
}

template<class T>
size_t HCudaMemory<T>::size() const
{
	return size_;
}

template<class T>
void HCudaMemory<T>::setZero()
{
	cuda(cudaMemset(data_.get(), 0, sizeof(T)*size_));
}

template<class T>
void HCudaMemory<T>::fromCPU(const std::vector<T>& c_data)
{
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaMemory::fromCPU: !valid()");
	}
	if (size_ != c_data.size()) {
		std::stringstream ss;
		ss << "HCudaMemory::fromCPU: m_size(" << size_ << ") != c_data.size(" << c_data.size() << ")";
		_raise(ss.str().c_str());
	}
#endif
	cuda(cudaMemcpy(data_.get(), c_data.data(), sizeof(T)*min(size_, c_data.size()), cudaMemcpyHostToDevice));
}

template<class T>
void HCudaMemory<T>::fromCPU(const std::vector<T>& c_data, int begin, int end) {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaMemory::fromCPU: !valid()");
	}
	if (size_ < end - begin) {
		_raise("HCudaMemory::fromCPU: m_size < end-begin");
	}
	if (begin < 0 || end > c_data.size()) {
		_raise("HCudaMemory::fromCPU: begin < 0 || end > c_data.size");
	}
#endif
	cuda(cudaMemcpy(data_.get(), c_data.data() + begin, sizeof(T)*(end - begin), cudaMemcpyHostToDevice));
}

template<class T>
void HCudaMemory<T>::fromGPU(T* data, size_t size, size_t offset)
{
	cuda(cudaMemcpy(data_.get() + offset, data, sizeof(T)*size, cudaMemcpyDeviceToDevice));
}



template<class T>
std::vector<T> HCudaMemory<T>::toCPU() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaMemory::toCPU: !valid()");
	}
#endif
	std::vector<T> res(size_);
	cuda(cudaMemcpy(res.data(), data_.get(), sizeof(T)*size_, cudaMemcpyDeviceToHost));
	return res;
}
template<class T>
void HCudaMemory<T>::toCPU(std::vector<T>& c_data) const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaMemory::toCPU: !valid()");
	}
	if (size_ != c_data.size()) {
		_raise("HCudaMemory::toCPU: m_size != c_data");
	}
#endif
	cuda(cudaMemcpy(c_data.data(), data_.get(), sizeof(T)*min(size_, c_data.size()), cudaMemcpyDeviceToHost));
}
template<class T>
void HCudaMemory<T>::toCPU(std::vector<T>& c_data, int begin, int end) const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaMemory::toCPU: !valid()");
	}
	if (size_ < end - begin) {
		_raise("HCudaMemory::toCPU: m_size < end-begin");
	}
	if (begin < 0 || end > c_data.size()) {
		_raise("HCudaMemory::toCPU: begin < 0 || end > c_data.size");
	}
#endif
	cuda(cudaMemcpy(c_data.data() + begin, data_.get(), sizeof(T)*(end - begin), cudaMemcpyDeviceToHost));
}

template<class T2>
std::ostream& operator<< (std::ostream& o, const HCudaMemory<T2>& h) {
	o << "HCudaMemory[size=" << h.size() << "][data=" << uint64_t(h.data()) << "]";
	return o;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudaMemory2D
///////////////////////////////////////////////////////////////////////////////////////////////////


template<class T>
HCudaMemory2D<T>::HCudaMemory2D()
{

}

template<class T>
HCudaMemory2D<T>::HCudaMemory2D(size_t nx, size_t ny, size_t pitch)
	:nx_(nx), ny_(ny), pitch_(pitch)
{
	T* raw;
	cuda(cudaMallocPitch(&raw, &pitch_, nx * sizeof(T), ny));
	pitch_ /= sizeof(T);
#if DEBUG_MEMORY
	std::cout << "HCudaMemory2D::HCudaMemory2D: raw(" << int(raw) << "), size(" << size << ")" << std::endl;
#endif
	data_ = std::shared_ptr<T>(raw, [](T* raw)
	{
		try
		{
#if DEBUG_MEMORY
			if (!raw) {
				_raise("HCudaMemory2D::deleter: raw is nullptr");
			}
			std::cout << "HCudaMemory2D::deleter: delete(" << int(raw) << ")" << std::endl;
#endif
			cuda(cudaFree(raw));
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}
	);
}

template<class T>
HCudaMemory2D<T>::HCudaMemory2D(size_t nx, size_t ny)
	:nx_(nx), ny_(ny)
{
	T* raw;
	cuda(cudaMallocPitch(&raw, &pitch_, nx * sizeof(T), ny));
	pitch_ /= sizeof(T);
#if DEBUG_MEMORY
	std::cout << "HCudaMemory2D::HCudaMemory2D: raw(" << int(raw) << "), size(" << size << ")" << std::endl;
#endif
	data_ = std::shared_ptr<T>(raw, [](T* raw)
	{
		try
		{
#if DEBUG_MEMORY
			if (!raw) {
				_raise("HCudaMemory2D::deleter: raw is nullptr");
			}
			std::cout << "HCudaMemory2D::deleter: delete(" << int(raw) << ")" << std::endl;
#endif
			cuda(cudaFree(raw));
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}
	);
}

template<class T>
HCudaMemory2D<T>::HCudaMemory2D(HCudaMemory2D<T> &data, bool is_released)
{
	init();
	nx_ = data.nx();
	ny_ = data.ny();
	pitch_ = data.pitch();

	data_ = std::shared_ptr<T>(data.data(), [is_released](T* raw) {
		try {
			if (is_released)
			{
#if DEBUG_MEMORY
				if (!raw) {
					_raise("HCudaMemory2D::deleter: raw is nullptr");
				}
				std::cout << "HCudaMemory2D::deleter: delete(" << raw << ")" << std::endl;
#endif
				cuda(cudaFree(raw));
			}
		}
		catch (std::exception& e) {
			std::cout << e.what() << std::endl;
		}
	});

}

template<class T>
HCudaMemory2D<T>::HCudaMemory2D(size_t nx, size_t ny, size_t pitch, T* data, bool is_released)
{
	init();
	nx_ = nx;
	ny_ = ny;
	pitch_ = pitch;

	data_ = std::shared_ptr<T>(data, [is_released](T* raw) {
		try {
			if (is_released)
			{
#if DEBUG_MEMORY
				if (!raw) {
					_raise("HCudaMemory2D::deleter: raw is nullptr");
				}
				std::cout << "HCudaMemory2D::deleter: delete(" << raw << ")" << std::endl;
#endif
				cuda(cudaFree(raw));
			}
		}
		catch (std::exception& e) {
			std::cout << e.what() << std::endl;
		}
	});
}


template<class T>
void HCudaMemory2D<T>::init()
{
	data_ = nullptr;
	nx_ = 0;
	ny_ = 0;
	pitch_ = 0;
}

template<class T>
bool HCudaMemory2D<T>::valid() const
{
	return data_ != nullptr;
}

template<class T>
T* HCudaMemory2D<T>::data() const
{
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaMemory::data: !valid()")
	}
#endif

	return data_.get();
}

template<class T>
size_t HCudaMemory2D<T>::nx() const
{
	return nx_;
}

template<class T>
size_t HCudaMemory2D<T>::ny() const
{
	return ny_;
}

template<class T>
size_t HCudaMemory2D<T>::pitch() const
{
	return pitch_;
}

template<class T>
void HCudaMemory2D<T>::setZero()
{
	cuda(cudaMemset(data_.get(), 0, sizeof(T)*pitch_*ny_));
}

template<class T>
void HCudaMemory2D<T>::fromCPU(const std::vector<T>& c_data)
{
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaMemory2D::fromCPU: !valid()");
	}
	if (nx_*ny_ != c_data.size()) {
		std::stringstream ss;
		ss << "HCudaMemory2D::fromCPU: size(" << nx_ * ny_ << ") != c_data.size(" << c_data.size() << ")";
		_raise(ss.str().c_str());
	}
#endif
	cuda(cudaMemcpy2D(data_.get(), sizeof(T)*pitch_, c_data.data(), sizeof(T)*nx_, sizeof(T)*nx_, ny_, cudaMemcpyHostToDevice));
}

template<class T>
void HCudaMemory2D<T>::fromCPU(const std::vector<T>& c_data, int begin, int end) {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaMemory2D::fromCPU: !valid()");
	}
	if (nx_*ny_ < end - begin) {
		_raise("HCudaMemory2D::fromCPU: size < end-begin");
	}
	if (begin < 0 || end > c_data.size()) {
		_raise("HCudaMemory2D::fromCPU: begin < 0 || end > c_data.size");
	}
#endif
	cuda(cudaMemcpy2D(data_.get(), sizeof(T)*pitch_, c_data.data() + begin, sizeof(T)*nx_, sizeof(T)*nx_, (end - begin) / nx_, cudaMemcpyHostToDevice));
}

template<class T>
void HCudaMemory2D<T>::fromGPU1D(T* data, size_t size, size_t offset)
{
	cuda(cudaMemcpy2D(data_.get() + offset, sizeof(T)*pitch_, data, sizeof(T)*nx_, sizeof(T)*nx_, ny_, cudaMemcpyDeviceToDevice));
}

template<class T>
void HCudaMemory2D<T>::fromGPU2D(HCudaMemory2D<T> &data)
{
	cuda(cudaMemcpy2D(data_.get(), sizeof(T)*pitch_, data.data(), sizeof(T)*pitch_, sizeof(T)*nx_, ny_, cudaMemcpyDeviceToDevice));
}

template<class T>
void HCudaMemory2D<T>::toGPU1D(T* data, size_t offset)
{
	cuda(cudaMemcpy2D(data, sizeof(T)*nx_, data_.get() + offset, sizeof(T)*pitch_, sizeof(T)*nx_, ny_, cudaMemcpyDeviceToDevice));
}


template<class T>
std::vector<T> HCudaMemory2D<T>::toCPU() const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaMemory::toCPU: !valid()");
	}
#endif
	std::vector<T> res(nx_*ny_);
	cuda(cudaMemcpy2D(res.data(), sizeof(T)*nx_, data_.get(), sizeof(T)*pitch_, sizeof(T)*nx_, ny_, cudaMemcpyDeviceToHost));
	return res;
}
template<class T>
void HCudaMemory2D<T>::toCPU(std::vector<T>& c_data) const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaMemory::toCPU: !valid()");
	}
	if (nx_*ny_ != c_data.size()) {
		_raise("HCudaMemory::toCPU: m_size != c_data");
	}
#endif
	cuda(cudaMemcpy2D(c_data.data(), sizeof(T)*nx_, data_.get(), sizeof(T)*pitch_, sizeof(T)*nx_, min(ny_, c_data.size() / nx_), cudaMemcpyDeviceToHost));
}
template<class T>
void HCudaMemory2D<T>::toCPU(std::vector<T>& c_data, int begin, int end) const {
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaMemory::toCPU: !valid()");
	}
	if (nx_*ny_ < end - begin) {
		_raise("HCudaMemory::toCPU: m_size < end-begin");
	}
	if (begin < 0 || end > c_data.size()) {
		_raise("HCudaMemory::toCPU: begin < 0 || end > c_data.size");
	}
#endif
	cuda(cudaMemcpy(c_data.data() + begin, sizeof(T)*nx_, data_.get(), sizeof(T)*pitch_, sizeof(T)*nx_, (end - begin) / nx_, cudaMemcpyDeviceToHost));
}

template<class T2>
std::ostream& operator<< (std::ostream& o, const HCudaMemory2D<T2>& h) {
	o << "HCudaMemory[size=" << h.nx()*h.ny() << "][data=" << h.data() << "]";
	return o;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudArray2DMemory
///////////////////////////////////////////////////////////////////////////////////////////////////


template<class T>
HCudaArray2DMemory<T>::HCudaArray2DMemory()
{

}

template<class T>
HCudaArray2DMemory<T>::HCudaArray2DMemory(size_t nx, size_t ny)
	:nx_(nx), ny_(ny)
{
	cudaArray* raw;
	channel_desc_ = cudaCreateChannelDesc<T>();
	cuda(cudaMallocArray(&raw, &channel_desc_, nx, ny));

#if DEBUG_MEMORY
	std::cout << "HCudaArray2DMemory::HCudaArray2DMemory: raw(" << int(raw) << "), size(" << nx << ", " << ny << ")" << std::endl;
#endif
	data_ = std::shared_ptr<cudaArray>(raw, [](cudaArray* raw)
	{
		try
		{
#if DEBUG_MEMORY
			if (!raw) {
				_raise("HCudaArray2DMemory::deleter: raw is nullptr");
			}
			std::cout << "HCudaArray2DMemory::deleter: delete(" << int(raw) << ")" << std::endl;
#endif
			cuda(cudaFreeArray(raw));
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}
	);
}


template<class T>
void HCudaArray2DMemory<T>::init()
{
	data_ = nullptr;
	nx_ = 0;
	ny_ = 0;
	channel_desc_ = { 0 };
}

template<class T>
bool HCudaArray2DMemory<T>::valid() const
{
	return data_ != nullptr;
}

template<class T>
cudaArray* HCudaArray2DMemory<T>::data() const
{
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaArray2DMemory::data: !valid()")
	}
#endif

	return data_.get();
}

template<class T>
size_t HCudaArray2DMemory<T>::nx() const
{
	return nx_;
}

template<class T>
size_t HCudaArray2DMemory<T>::ny() const
{
	return ny_;
}

template<class T>
void HCudaArray2DMemory<T>::fromCPU(const std::vector<T>& c_data)
{
	cuda(cudaMemcpyToArray(data_.get(), 0, 0, c_data.data(), nx_*ny_ * sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
void HCudaArray2DMemory<T>::fromGPU(T* g_data)
{
	cuda(cudaMemcpyToArray(data_.get(), 0, 0, g_data, nx_*ny_ * sizeof(float), cudaMemcpyDeviceToDevice));
}

template<class T>
std::ostream& operator<< (std::ostream& o, const HCudaArray2DMemory<T>& h)
{
	o << "HCudaArray2DMemory[size=" << h.nx() << "," << h.ny() << "][data=" << h.data() << "]";
	return o;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudaArray3DMemory
///////////////////////////////////////////////////////////////////////////////////////////////////


template<class T>
HCudaArray3DMemory<T>::HCudaArray3DMemory()
{

}

template<class T>
HCudaArray3DMemory<T>::HCudaArray3DMemory(size_t nx, size_t ny, size_t nz, bool isLayered)
{
	size_ = make_cudaExtent(nx, ny, nz);
	cudaArray* raw;
	channel_desc_ = cudaCreateChannelDesc<T>();
	if (isLayered)
	{
		cuda(cudaMalloc3DArray(&raw, &channel_desc_, size_, cudaArrayLayered));
	}
	else
	{
		cuda(cudaMalloc3DArray(&raw, &channel_desc_, size_));
	}

#if DEBUG_MEMORY
	std::cout << "HCudaArray3DMemory::HCudaArray3DMemory: raw(" << int(raw) << "), size(" << nx << "," << ny << "," << nz << ")" << std::endl;
#endif


	data_ = std::shared_ptr<cudaArray>(raw, [](cudaArray* raw)
	{
		try
		{
#if DEBUG_MEMORY
			if (!raw) {
				_raise("HCudaArray3DMemory::deleter: raw is nullptr");
			}
			std::cout << "HCudaArray3DMemory::deleter: delete(" << int(raw) << ")" << std::endl;
#endif
			cuda(cudaFreeArray(raw));
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}
	);

	copy_params_.extent = size_;
	copy_params_.dstArray = data_.get();
}


template<class T>
void HCudaArray3DMemory<T>::init()
{
	data_ = nullptr;
	size_ = { 0 };
	channel_desc_ = { 0 };
}

template<class T>
bool HCudaArray3DMemory<T>::valid() const
{
	return data_ != nullptr;
}

template<class T>
cudaArray* HCudaArray3DMemory<T>::data() const
{
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaArray2DMemory::data: !valid()")
	}
#endif
	return data_.get();
}

template<class T>
cudaExtent HCudaArray3DMemory<T>::size() const
{
	return size_;
}

template<class T>
void HCudaArray3DMemory<T>::fromCPU(const std::vector<T>& c_data)
{
	copy_params_.srcPtr = make_cudaPitchedPtr((void *)c_data.data(), sizeof(T)*size_.width, size_.width, size_.height);
	copy_params_.kind = cudaMemcpyHostToDevice;
	cuda(cudaMemcpy3D(&copy_params_));
}

template<class T>
void HCudaArray3DMemory<T>::fromGPU(T* g_data)
{
	copy_params_.srcPtr = make_cudaPitchedPtr((void *)g_data, sizeof(T)*size_.width, size_.width, size_.height);
	copy_params_.kind = cudaMemcpyDeviceToDevice;
	cuda(cudaMemcpy3D(&copy_params_));
}

template<class T>
cudaMemcpy3DParms* HCudaArray3DMemory<T>::get_copy_params()
{
	return &copy_params_;
}

template<class T2>
std::ostream& operator<< (std::ostream& o, const HCudaArray3DMemory<T2>& h)
{
	o << "HCudaArray2DMemory[size=" << h.size().width << "," << h.size().height << "," << h.size().depth << "][data=" << h.data() << "]";
	return o;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// @HCudaTextureObject
///////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
HCudaTextureObject<T>::HCudaTextureObject()
{
	init();
}

template<class T>
HCudaTextureObject<T>::HCudaTextureObject(size_t nx, size_t ny, cudaTextureReadMode read_mode, cudaTextureFilterMode filter_mode, std::vector<cudaTextureAddressMode> address_mode)
{
	init();
	array2D_ = HCudaArray2DMemory<T>(nx, ny);
	res_desc_.resType = cudaResourceTypeArray;
	res_desc_.res.array.array = array2D_.data();

	tex_desc_.normalizedCoords = false;
	tex_desc_.filterMode = filter_mode;
	tex_desc_.readMode = read_mode;

	for (int i = 0; i < address_mode.size(); i++)
	{
		tex_desc_.addressMode[i] = address_mode[i];
	}

	cudaTextureObject_t *raw = new cudaTextureObject_t();
	cuda(cudaCreateTextureObject(raw, &res_desc_, &tex_desc_, NULL));

#if DEBUG_MEMORY
	std::cout << "HCudaTextureObject::HCudaTextureObject: raw(" << int(raw) << "), size(" << nx << "," << ny << ")" << std::endl;
#endif

	data_ = std::shared_ptr<cudaTextureObject_t>(raw, [](cudaTextureObject_t* raw)
	{
		try
		{
#if DEBUG_MEMORY
			if (!raw) {
				_raise("HCudaTextureObject::deleter: raw is nullptr");
			}
			std::cout << "HCudaTextureObject::deleter: delete(" << int(raw) << ")" << std::endl;
#endif
			cuda(cudaDestroyTextureObject(*raw));
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}
	);

}

template<class T>
HCudaTextureObject<T>::HCudaTextureObject(HCudaMemory2D<T> &mem, cudaTextureReadMode read_mode, cudaTextureFilterMode filter_mode, std::vector<cudaTextureAddressMode> address_mode)
{
	init();
	mem2D_ = mem;
	res_desc_.resType = cudaResourceTypePitch2D;
	res_desc_.res.pitch2D.devPtr = mem2D_.data();
	res_desc_.res.pitch2D.width = mem2D_.nx();
	res_desc_.res.pitch2D.height = mem2D_.ny();
	res_desc_.res.pitch2D.pitchInBytes = mem2D_.pitch() * sizeof(T);
	res_desc_.res.pitch2D.desc = cudaCreateChannelDesc<T>();

	tex_desc_.normalizedCoords = false;
	tex_desc_.filterMode = filter_mode;
	tex_desc_.readMode = read_mode;

	for (int i = 0; i < address_mode.size(); i++)
	{
		tex_desc_.addressMode[i] = address_mode[i];
	}

	cudaTextureObject_t *raw = new cudaTextureObject_t();
	cuda(cudaCreateTextureObject(raw, &res_desc_, &tex_desc_, NULL));

#if DEBUG_MEMORY
	std::cout << "HCudaTextureObject::HCudaTextureObject: raw(" << int(raw) << "), size(" << array2D_.nx() << "," << array2D_.ny() << ")" << std::endl;
#endif

	data_ = std::shared_ptr<cudaTextureObject_t>(raw, [](cudaTextureObject_t* raw)
	{
		try
		{
#if DEBUG_MEMORY
			if (!raw) {
				_raise("HCudaTextureObject::deleter: raw is nullptr");
			}
			std::cout << "HCudaTextureObject::deleter: delete(" << int(raw) << ")" << std::endl;
#endif
			cuda(cudaDestroyTextureObject(*raw));
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}
	);

}

template<class T>
HCudaTextureObject<T>::HCudaTextureObject(HCudaArray2DMemory<T> arr, cudaTextureReadMode read_mode, cudaTextureFilterMode filter_mode, std::vector<cudaTextureAddressMode> address_mode)
{
	init();
	array2D_ = arr;
	res_desc_.resType = cudaResourceTypeArray;
	res_desc_.res.array.array = array2D_.data();

	tex_desc_.normalizedCoords = false;
	tex_desc_.filterMode = filter_mode;
	tex_desc_.readMode = read_mode;

	for (int i = 0; i < address_mode.size(); i++)
	{
		tex_desc_.addressMode[i] = address_mode[i];
	}

	cudaTextureObject_t *raw = new cudaTextureObject_t();
	cuda(cudaCreateTextureObject(raw, &res_desc_, &tex_desc_, NULL));

#if DEBUG_MEMORY
	std::cout << "HCudaTextureObject::HCudaTextureObject: raw(" << int(raw) << "), size(" << array2D_.nx() << "," << array2D_.ny() << ")" << std::endl;
#endif

	data_ = std::shared_ptr<cudaTextureObject_t>(raw, [](cudaTextureObject_t* raw)
	{
		try
		{
#if DEBUG_MEMORY
			if (!raw) {
				_raise("HCudaTextureObject::deleter: raw is nullptr");
			}
			std::cout << "HCudaTextureObject::deleter: delete(" << int(raw) << ")" << std::endl;
#endif
			cuda(cudaDestroyTextureObject(*raw));
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}
	);

}

template<class T>
HCudaTextureObject<T>::HCudaTextureObject(size_t nx, size_t ny, size_t nz, cudaTextureReadMode read_mode, cudaTextureFilterMode filter_mode, std::vector<cudaTextureAddressMode> address_mode, bool isLayered)
{
	init();
	array3D_ = HCudaArray3DMemory<T>(nx, ny, nz, isLayered);
	res_desc_.resType = cudaResourceTypeArray;
	res_desc_.res.array.array = array3D_.data();

	tex_desc_.normalizedCoords = false;
	tex_desc_.filterMode = filter_mode;
	tex_desc_.readMode = read_mode;

	for (int i = 0; i < address_mode.size(); i++)
	{
		tex_desc_.addressMode[i] = address_mode[i];
	}

	cudaTextureObject_t *raw = new cudaTextureObject_t();
	cuda(cudaCreateTextureObject(raw, &res_desc_, &tex_desc_, NULL));

#if DEBUG_MEMORY
	std::cout << "HCudaTextureObject::HCudaTextureObject: raw(" << int(raw) << "), size(" << nx << "," << ny << "," << nz << ")" << std::endl;
#endif

	data_ = std::shared_ptr<cudaTextureObject_t>(raw, [](cudaTextureObject_t* raw)
	{
		try
		{
#if DEBUG_MEMORY
			if (!raw) {
				_raise("HCudaTextureObject::deleter: raw is nullptr");
			}
			std::cout << "HCudaTextureObject::deleter: delete(" << int(raw) << ")" << std::endl;
#endif
			cuda(cudaDestroyTextureObject(*raw));
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}
	);
}

template<class T>
HCudaTextureObject<T>::HCudaTextureObject(HCudaArray3DMemory<T> arr, cudaTextureReadMode read_mode, cudaTextureFilterMode filter_mode, std::vector<cudaTextureAddressMode> address_mode)
{
	init();
	array3D_ = arr;
	res_desc_.resType = cudaResourceTypeArray;
	res_desc_.res.array.array = array3D_.data();

	tex_desc_.normalizedCoords = false;
	tex_desc_.filterMode = filter_mode;
	tex_desc_.readMode = read_mode;

	for (int i = 0; i < address_mode.size(); i++)
	{
		tex_desc_.addressMode[i] = address_mode[i];
	}

	cudaTextureObject_t *raw = new cudaTextureObject_t();
	cuda(cudaCreateTextureObject(raw, &res_desc_, &tex_desc_, NULL));
#if DEBUG_MEMORY
	std::cout << "HCudaTextureObject::HCudaTextureObject: raw(" << int(raw) << "), size(" << arr.size().width << "," << arr.size().height << "," << arr.size().depth << ")" << std::endl;
#endif

	data_ = std::shared_ptr<cudaTextureObject_t>(raw, [](cudaTextureObject_t* raw)
	{
		try
		{
#if DEBUG_MEMORY
			if (!raw) {
				_raise("HCudaTextureObject::deleter: raw is nullptr");
			}
			std::cout << "HCudaTextureObject::deleter: delete(" << int(raw) << ")" << std::endl;
#endif
			cuda(cudaDestroyTextureObject(*raw));
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}
	);

}


template<class T>
void HCudaTextureObject<T>::init()
{
	array2D_.init();
	array3D_.init();
	mem2D_.init();
	data_ = nullptr;
	memset(&res_desc_, 0, sizeof(res_desc_));
	memset(&tex_desc_, 0, sizeof(tex_desc_));
}

template<class T>
bool HCudaTextureObject<T>::valid() const
{
	return data_ != nullptr;
}

template<class T>
cudaTextureObject_t HCudaTextureObject<T>::data() const
{
#if DEBUG_MEMORY
	if (!valid()) {
		_raise("HCudaArray2DMemory::data: !valid()")
	}
#endif
	return *data_;
}

template<class T>
HCudaMemory2D<T> HCudaTextureObject<T>::mem2D() const
{
	return mem2D_;
}


template<class T>
void HCudaTextureObject<T>::fromCPUto2Dmem(const std::vector<T>& c_data)
{
	mem2D_.fromCPU(c_data);
}

template<class T>
void HCudaTextureObject<T>::fromGPUto2Dmem(T* g_data)
{
	mem2D_.fromGPU1D(g_data);
}


template<class T>
void HCudaTextureObject<T>::fromCPUto2D(const std::vector<T>& c_data)
{
	cuda(cudaMemcpyToArray(array2D_.data(), 0, 0, c_data.data(), array2D_.nx()*array2D_.ny() * sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
void HCudaTextureObject<T>::fromGPUto2D(T* g_data)
{
	cuda(cudaMemcpyToArray(array2D_.data(), 0, 0, g_data, array2D_.nx()*array2D_.ny() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template<class T>
void HCudaTextureObject<T>::fromCPUto3D(const std::vector<T>& c_data)
{
	cudaExtent size = array3D_.size();
	array3D_.get_copy_params()->srcPtr = make_cudaPitchedPtr((void *)c_data.data(), sizeof(T)*size.width, size.width, size.height);
	array3D_.get_copy_params()->kind = cudaMemcpyHostToDevice;
	cuda(cudaMemcpy3D(array3D_.get_copy_params()));
}

template<class T>
void HCudaTextureObject<T>::fromGPUto3D(T* g_data)
{
	cudaExtent size = array3D_.size();
	array3D_.get_copy_params()->srcPtr = make_cudaPitchedPtr((void *)g_data, sizeof(T)*size.width, size.width, size.height);
	array3D_.get_copy_params()->kind = cudaMemcpyDeviceToDevice;
	cuda(cudaMemcpy3D(array3D_.get_copy_params()));
}


template<class T2>
std::ostream& operator<< (std::ostream& o, const HCudaTextureObject<T2>& h) {
	o << "HCudaTextureObject[data=" << h.data() << "]";
	return o;
}