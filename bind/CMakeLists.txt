cmake_minimum_required(VERSION 3.0.0)
project(bind LANGUAGES CXX CUDA)

# -------------------------------------------------------- #
# compile option
# -------------------------------------------------------- #
if(WIN32)
elseif(UNIX)
	set(CMAKE_CXX_STANDARD 14)
	set(CMAKE_CXX_STANDARD)
	add_compile_options(-std=c++14 -Wall -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -fno-sized-deallocation -fopenmp)
endif()

# -------------------------------------------------------- #
# target
# -------------------------------------------------------- #
if(WIN32)
elseif(UNIX)
	set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:$ORIGIN") # target 앞에서 해야 의미있음. install해서 복사될때 적용됨.
endif()

add_library(${PROJECT_NAME} SHARED cout_overloading.cpp cuda_global_setting.cpp cudnn_handle.cpp Tensor4d.cpp bind.cpp)
# add_executable(${PROJECT_NAME} cout_overloading.cpp cuda_global_setting.cpp cudnn_handle.cpp Tensor4d.cpp bind.cpp)

if(WIN3)
elseif(UNIX)
	set_target_properties(${PROJECT_NAME} PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
	)
endif()

install( 
	TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)

# -------------------------------------------------------- #
# library setting
# -------------------------------------------------------- #
if(WIN32)
	
elseif(UNIX)
	target_include_directories(${PROJECT_NAME} PRIVATE
		"../"
		"./"
		# "/home/hosan/anaconda3/include/python3.7m"
		# "/home/hosan/anaconda3/lib/python3.7/site-packages/torch"
		# "/home/hosan/anaconda3/lib/python3.7/site-packages/torch/include"
		# "/home/hosan/anaconda3/lib/python3.7/site-packages/torch/include/torch/csrc/api/include"
		# "/home/hosan/anaconda3/lib/python3.7/site-packages/torch/include/TH"
		# "/home/hosan/anaconda3/lib/python3.7/site-packages/torch/include/THC"
		
		"/home/hosan/anaconda3/envs/fastmri/include/python3.8"
		"/home/hosan/anaconda3/envs/fastmri/lib/python3.8/site-packages/torch"
		"/home/hosan/anaconda3/envs/fastmri/lib/python3.8/site-packages/torch/include"
		"/home/hosan/anaconda3/envs/fastmri/lib/python3.8/site-packages/torch/include/torch/csrc/api/include"
		"/home/hosan/anaconda3/envs/fastmri/lib/python3.8/site-packages/torch/include/TH"
		"/home/hosan/anaconda3/envs/fastmri/lib/python3.8/site-packages/torch/include/THC"
		"/usr/local/cuda/include"
	)
	target_link_directories(${PROJECT_NAME} PRIVATE
		"/home/hosan/anaconda3/envs/fastmri/lib"
		"/home/hosan/anaconda3/envs/fastmri/lib/python3.8/site-packages/torch/lib"
		"/usr/local/cuda/lib64"
	)
	target_link_libraries(${PROJECT_NAME} PRIVATE
		cudart
		cudnn
		cublas
		c10
		c10_cuda
		caffe2_nvrtc
		torch
		torch_cpu
		torch_cuda
		torch_python
		python3.8
		dl
	)
endif()


