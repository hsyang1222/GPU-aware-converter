#include "cuda_global_setting.h"

using namespace std;

CudaSetDeviceManager::CudaSetDeviceManager() {

}

void CudaSetDeviceManager::push(int id) {
	// 현재 device id 저장
	int current_device_id;
	cuda(cudaGetDevice(&current_device_id));
	id_olds_.push_back(current_device_id);

	//vector<int> current_device_id(1);
	//cuda(cudaGetDevice(current_device_id.data()));
	//id_olds_.push_back(current_device_id[0]);

	// 주어진 id로 set
	cuda(cudaSetDevice(id));
}

void CudaSetDeviceManager::pop() {
	if (id_olds_.size() == 0) {
		throw std::runtime_error("CudaSetDeviceManager: cannot pop");
	}
	
	int id = id_olds_.back();
	id_olds_.pop_back();
	cuda(cudaSetDevice(id));
}