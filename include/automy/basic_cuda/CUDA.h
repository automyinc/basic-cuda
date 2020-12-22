/*
 * CUDA.h
 *
 *  Created on: Jan 13, 2018
 *      Author: mad
 */

#ifndef INCLUDE_BASIC_CUDA_H_
#define INCLUDE_BASIC_CUDA_H_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cmath>
#include <stdexcept>

#define __scalar__ if((threadIdx.x | threadIdx.y | threadIdx.z) == 0)

typedef long long int int64_cu;
typedef unsigned long long int uint64_cu;


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

__device__ __host__ inline float2 operator+(const float2 a, const float2 b) {
	float2 res;
	res.x = a.x + b.x;
	res.y = a.y + b.y;
	return res;
}

__device__ __host__ inline float2 operator-(const float2 a, const float2 b) {
	float2 res;
	res.x = a.x - b.x;
	res.y = a.y - b.y;
	return res;
}

__device__ __host__ inline float2 operator*(const float2 a, const float b) {
	float2 res;
	res.x = a.x * b;
	res.y = a.y * b;
	return res;
}

__device__ __host__ inline float2 operator/(const float2 a, const float b) {
	float2 res;
	res.x = a.x / b;
	res.y = a.y / b;
	return res;
}

__device__ __host__ inline double2 operator+(const double2 a, const double2 b) {
	double2 res;
	res.x = a.x + b.x;
	res.y = a.y + b.y;
	return res;
}

__device__ __host__ inline double2 operator-(const double2 a, const double2 b) {
	double2 res;
	res.x = a.x - b.x;
	res.y = a.y - b.y;
	return res;
}

__device__ __host__ inline double2 operator*(const double2 a, const double b) {
	double2 res;
	res.x = a.x * b;
	res.y = a.y * b;
	return res;
}

__device__ __host__ inline double2 operator/(const double2 a, const double b) {
	double2 res;
	res.x = a.x / b;
	res.y = a.y / b;
	return res;
}

__device__ __host__ inline float3 operator+(const float3 a, const float3 b) {
	float3 res;
	res.x = a.x + b.x;
	res.y = a.y + b.y;
	res.z = a.z + b.z;
	return res;
}

__device__ __host__ inline float3 operator-(const float3 a, const float3 b) {
	float3 res;
	res.x = a.x - b.x;
	res.y = a.y - b.y;
	res.z = a.z - b.z;
	return res;
}

__device__ __host__ inline float3 operator*(const float3 a, const float b) {
	float3 res;
	res.x = a.x * b;
	res.y = a.y * b;
	res.z = a.z * b;
	return res;
}

__device__ __host__ inline float3 operator/(const float3 a, const float b) {
	float3 res;
	res.x = a.x / b;
	res.y = a.y / b;
	res.z = a.z / b;
	return res;
}

__device__ __host__ inline double3 operator+(const double3 a, const double3 b) {
	double3 res;
	res.x = a.x + b.x;
	res.y = a.y + b.y;
	res.z = a.z + b.z;
	return res;
}

__device__ __host__ inline double3 operator-(const double3 a, const double3 b) {
	double3 res;
	res.x = a.x - b.x;
	res.y = a.y - b.y;
	res.z = a.z - b.z;
	return res;
}

__device__ __host__ inline double3 operator*(const double3 a, const double b) {
	double3 res;
	res.x = a.x * b;
	res.y = a.y * b;
	res.z = a.z * b;
	return res;
}

__device__ __host__ inline double3 operator/(const double3 a, const double b) {
	double3 res;
	res.x = a.x / b;
	res.y = a.y / b;
	res.z = a.z / b;
	return res;
}

__device__ __host__ inline float4 operator+(const float4 a, const float4 b) {
	float4 res;
	res.x = a.x + b.x;
	res.y = a.y + b.y;
	res.z = a.z + b.z;
	res.w = a.w + b.w;
	return res;
}

__device__ __host__ inline float4 operator-(const float4 a, const float4 b) {
	float4 res;
	res.x = a.x - b.x;
	res.y = a.y - b.y;
	res.z = a.z - b.z;
	res.w = a.w - b.w;
	return res;
}

__device__ __host__ inline float4 operator*(const float4 a, const float b) {
	float4 res;
	res.x = a.x * b;
	res.y = a.y * b;
	res.z = a.z * b;
	res.w = a.w * b;
	return res;
}

__device__ __host__ inline float4 operator/(const float4 a, const float b) {
	float4 res;
	res.x = a.x / b;
	res.y = a.y / b;
	res.z = a.z / b;
	res.w = a.w / b;
	return res;
}

__device__ __host__ inline uint4 operator+(const uint4 a, const uchar4 b) {
	uint4 res;
	res.x = a.x + b.x;
	res.y = a.y + b.y;
	res.z = a.z + b.z;
	res.w = a.w + b.w;
	return res;
}

__device__ __host__ inline uint4 operator+(const uint4 a, const uint4 b) {
	uint4 res;
	res.x = a.x + b.x;
	res.y = a.y + b.y;
	res.z = a.z + b.z;
	res.w = a.w + b.w;
	return res;
}

__device__ __host__ inline uint4 operator*(const uint4 a, const unsigned int b) {
	uint4 res;
	res.x = a.x * b;
	res.y = a.y * b;
	res.z = a.z * b;
	res.w = a.w * b;
	return res;
}

__device__ __host__ inline float dot(const float2 a, const float2 b) {
	return a.x * b.x + a.y * b.y;
}

__device__ __host__ inline float dot(const float3 a, const float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ inline float dot(const float4 a, const float4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __host__ inline float square_norm(const float2 a) {
	return (a.x*a.x + a.y*a.y);
}

__device__ __host__ inline float square_norm(const float3 a) {
	return (a.x*a.x + a.y*a.y + a.z*a.z);
}

__device__ __host__ inline float square_norm(const float4 a) {
	return (a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w);
}

__device__ __host__ inline float norm(const float2 a) {
	return sqrtf(a.x*a.x + a.y*a.y);
}

__device__ __host__ inline float norm(const float3 a) {
	return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

__device__ __host__ inline float norm(const float4 a) {
	return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w);
}

__device__ __host__ inline float2 mul_22_2(const float* mat, const float2 b) {
	float2 res;
	res.x = mat[0] * b.x + mat[2] * b.y;
	res.y = mat[1] * b.x + mat[3] * b.y;
	return res;
}

__device__ __host__ inline void get_rotate2(float* mat, const float radians) {
	mat[0] = cosf(radians);
	mat[1] = sinf(radians);
	mat[2] = -sinf(radians);
	mat[3] = cosf(radians);
}

__device__ __host__ inline float2 rotate2(const float2 b, const float radians) {
	float mat[4];
	get_rotate2(mat, radians);
	return mul_22_2(mat, b);
}

__device__ __host__ inline float3 mul_33_3(const float* mat, const float3 b) {
	float3 res;
	res.x = mat[0] * b.x + mat[3] * b.y + mat[6] * b.z;
	res.y = mat[1] * b.x + mat[4] * b.y + mat[7] * b.z;
	res.z = mat[2] * b.x + mat[5] * b.y + mat[8] * b.z;
	return res;
}

__device__ __host__ inline int3 mul_33_3(const int* mat, const uchar4 b) {
	int3 res;
	res.x = mat[0] * b.x + mat[3] * b.y + mat[6] * b.z;
	res.y = mat[1] * b.x + mat[4] * b.y + mat[7] * b.z;
	res.z = mat[2] * b.x + mat[5] * b.y + mat[8] * b.z;
	return res;
}

__device__ __host__ inline int3 mul_33_3(const int* mat, const short3 b) {
	int3 res;
	res.x = mat[0] * b.x + mat[3] * b.y + mat[6] * b.z;
	res.y = mat[1] * b.x + mat[4] * b.y + mat[7] * b.z;
	res.z = mat[2] * b.x + mat[5] * b.y + mat[8] * b.z;
	return res;
}

__device__ __host__ inline float3 mul_34_3(const float* mat, const float3 b) {
	float3 res;
	res.x = mat[0] * b.x + mat[3] * b.y + mat[6] * b.z + mat[9];
	res.y = mat[1] * b.x + mat[4] * b.y + mat[7] * b.z + mat[10];
	res.z = mat[2] * b.x + mat[5] * b.y + mat[8] * b.z + mat[11];
	return res;
}

__device__ __host__ inline double3 mul_34_3(const double* mat, const double3 b) {
	double3 res;
	res.x = mat[0] * b.x + mat[3] * b.y + mat[6] * b.z + mat[9];
	res.y = mat[1] * b.x + mat[4] * b.y + mat[7] * b.z + mat[10];
	res.z = mat[2] * b.x + mat[5] * b.y + mat[8] * b.z + mat[11];
	return res;
}

template<typename T>
__device__ __host__ inline void mul_34_34(T* Y, const T* A, const T* B) {
	Y[0] =  A[0]*B[0] + A[3]*B[1] +  A[6]*B[2];
	Y[3] =  A[0]*B[3] + A[3]*B[4] +  A[6]*B[5];
	Y[6] =  A[0]*B[6] + A[3]*B[7] +  A[6]*B[8];
	Y[9] =  A[0]*B[9] + A[3]*B[10] + A[6]*B[11] + A[9];
	
	Y[1] =  A[1]*B[0] + A[4]*B[1] +  A[7]*B[2];
	Y[4] =  A[1]*B[3] + A[4]*B[4] +  A[7]*B[5];
	Y[7] =  A[1]*B[6] + A[4]*B[7] +  A[7]*B[8];
	Y[10] = A[1]*B[9] + A[4]*B[10] + A[7]*B[11] + A[10];
	
	Y[2] =  A[2]*B[0] + A[5]*B[1] +  A[8]*B[2];
	Y[5] =  A[2]*B[3] + A[5]*B[4] +  A[8]*B[5];
	Y[8] =  A[2]*B[6] + A[5]*B[7] +  A[8]*B[8];
	Y[11] = A[2]*B[9] + A[5]*B[10] + A[8]*B[11] + A[11];
}

/*
 * Matrix multiplication assuming row major storage: Y_NK = A_NM * B_MK
 */
template<int N, int M, int K, typename T>
__device__ __host__ inline void mul_NM_K(T* Y, const T* A, const T* B) {
	for(int i = 0; i < N*K; ++i) {
		Y[i] = 0;
	}
	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < K; ++j) {
			for(int k = 0; k < M; ++k) {
				Y[j*N + i] += A[k*N + i] * B[j*M + k];
			}
		}
	}
}

/*
 * Matrix multiplication assuming row major storage: Y_MK = A_NM^T * B_MK
 */
template<int N, int M, int K, typename T>
__device__ __host__ inline void mul_NM_T_K(T* Y, const T* A, const T* B) {
	for(int i = 0; i < M*K; ++i) {
		Y[i] = 0;
	}
	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < K; ++j) {
			for(int k = 0; k < N; ++k) {
				Y[j*M + i] += A[i*N + k] * B[j*N + k];
			}
		}
	}
}

template<typename T>
__device__ __host__ inline void inverse_33(T* A_inv, const T* A) {
	const T det = 	A[0 + 0 * 3] * (A[1 + 1 * 3] * A[2 + 2 * 3] - A[2 + 1 * 3] * A[1 + 2 * 3]) -
					A[0 + 1 * 3] * (A[1 + 0 * 3] * A[2 + 2 * 3] - A[1 + 2 * 3] * A[2 + 0 * 3]) +
					A[0 + 2 * 3] * (A[1 + 0 * 3] * A[2 + 1 * 3] - A[1 + 1 * 3] * A[2 + 0 * 3]);
	const T inv_det = 1.f / det;
	A_inv[0 + 0 * 3] = (A[1 + 1 * 3] * A[2 + 2 * 3] - A[2 + 1 * 3] * A[1 + 2 * 3]) * inv_det;
	A_inv[0 + 1 * 3] = (A[0 + 2 * 3] * A[2 + 1 * 3] - A[0 + 1 * 3] * A[2 + 2 * 3]) * inv_det;
	A_inv[0 + 2 * 3] = (A[0 + 1 * 3] * A[1 + 2 * 3] - A[0 + 2 * 3] * A[1 + 1 * 3]) * inv_det;
	A_inv[1 + 0 * 3] = (A[1 + 2 * 3] * A[2 + 0 * 3] - A[1 + 0 * 3] * A[2 + 2 * 3]) * inv_det;
	A_inv[1 + 1 * 3] = (A[0 + 0 * 3] * A[2 + 2 * 3] - A[0 + 2 * 3] * A[2 + 0 * 3]) * inv_det;
	A_inv[1 + 2 * 3] = (A[1 + 0 * 3] * A[0 + 2 * 3] - A[0 + 0 * 3] * A[1 + 2 * 3]) * inv_det;
	A_inv[2 + 0 * 3] = (A[1 + 0 * 3] * A[2 + 1 * 3] - A[2 + 0 * 3] * A[1 + 1 * 3]) * inv_det;
	A_inv[2 + 1 * 3] = (A[2 + 0 * 3] * A[0 + 1 * 3] - A[0 + 0 * 3] * A[2 + 1 * 3]) * inv_det;
	A_inv[2 + 2 * 3] = (A[0 + 0 * 3] * A[1 + 1 * 3] - A[1 + 0 * 3] * A[0 + 1 * 3]) * inv_det;
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
