#pragma once

#define _USE_MATH_DEFINES
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "helper_cuda.h"
#include "helper_math.h"
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////////////////////////
/// cuda err
///////////////////////////////////////////////////////////////////////////////////////////////////

#define cuda(x)\
if ((x) != cudaSuccess && (x) != CUBLAS_STATUS_SUCCESS){\
	std::stringstream ss; \
	ss << "## cuda err at " << __FILE__ << "(" << __LINE__ << "):\n\t@code: " << #x << "\n\t@err: " << _cudaGetErrorEnum(x) << std::endl; \
	throw std::runtime_error(ss.str().c_str()); \
}

#define _raise(x)\
{\
	std::stringstream _ss;\
	_ss << __FILE__ << " (" << __LINE__ << "): " << x; \
	throw std::runtime_error(_ss.str().c_str());\
}

class CudaSetDeviceManager {
public:
	CudaSetDeviceManager();
	void push(int id);
	void pop();

protected:
	std::vector<int> id_olds_;
};