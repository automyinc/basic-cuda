/*
 * CUDA_device.h
 *
 *  Created on: Jan 13, 2018
 *      Author: mad
 */

#ifndef INCLUDE_BASIC_CUDA_DEVICE_H_
#define INCLUDE_BASIC_CUDA_DEVICE_H_

#include <automy/basic_cuda/CUDA.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define __scalar__ if((threadIdx.x | threadIdx.y | threadIdx.z) == 0)


template<typename T>
__device__ inline T abs(const T a) {
	return a >= 0 ? a : -a;
}

template<typename T>
__device__ inline T min(const T a, const T b) {
	return a < b ? a : b;
}

template<typename T>
__device__ inline T max(const T a, const T b) {
	return a > b ? a : b;
}

__device__
inline half2 uint_as_half2(const uint v) {
	return *((const half2*)&v);
}

__device__
inline uint half2_as_uint(const half2 v) {
	return *((const uint*)&v);
}

__device__
inline char4 uint_as_char4(const uint& v) {
	return *((const char4*)&v);
}

__device__
inline uint char4_as_uint(const char4& v) {
	return *((const uint*)&v);
}

template<typename T>
__device__ inline T fetch_xy(int x, int y, const T* data, const int width)
{
	return __ldg(&data[y * width + x]);
}

template<typename T>
__device__ inline T fetch_zxy(int x, int y, int z, const T* data, const int width, const int depth)
{
	return __ldg(&data[(y * width + x) * depth + z]);
}

template<typename T>
__device__ inline T fetch_xyz(int x, int y, int z, const T* data, const int width, const int height)
{
	return __ldg(&data[(z * height + y) * width + x]);
}

template<typename T>
__device__ inline T fetch_uv(float u, float v, cudaTextureObject_t texture)
{
	return tex2D<T>(texture, u, v);
}

template<typename T>
__device__ inline T fetch_xy(float x, float y, cudaTextureObject_t texture, const float inv_width, const float inv_height)
{
	return tex2D<T>(texture, (x + 0.5f) * inv_width, (y + 0.5f) * inv_height);
}

template<typename T>
__device__ inline void warp_sum_32(T& value)
{
	for(int k = 16; k >= 1; k /= 2) {
		value += __shfl_xor_sync(0xffffffff, value, k, 32);
	}
}


#endif /* INCLUDE_BASIC_CUDA_DEVICE_H_ */
