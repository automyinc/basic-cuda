/*
 * CUDA.h
 *
 *  Created on: Jan 13, 2018
 *      Author: mad
 */

#ifndef INCLUDE_BASIC_CUDA_H_
#define INCLUDE_BASIC_CUDA_H_

#include <cuda_runtime_api.h>

#include <cmath>
#include <stdexcept>


inline void cuda_check(const cudaError_t& code) {
	if(code != cudaSuccess) {
		throw std::runtime_error(std::string(cudaGetErrorString(code)));
	}
}

inline void cuda_check(const cudaError_t& code, const std::string& message) {
	if(code != cudaSuccess) {
		throw std::runtime_error(message + std::string(cudaGetErrorString(code)));
	}
}

inline dim3 ceiled_grid_dim(uint32_t width, uint32_t height, dim3 block_dim) {
	return dim3((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);
}

inline dim3 ceiled_grid_dim(uint32_t width, uint32_t height, uint32_t depth, dim3 block_dim) {
	return dim3((width + block_dim.x - 1) / block_dim.x,
				(height + block_dim.y - 1) / block_dim.y,
				(depth + block_dim.z - 1) / block_dim.z);
}


#endif /* INCLUDE_BASIC_CUDA_H_ */
