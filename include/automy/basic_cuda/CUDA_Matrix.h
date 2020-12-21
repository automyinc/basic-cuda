/*
 * CUDA_Matrix.h
 *
 *  Created on: Feb 10, 2018
 *      Author: dev
 */

#ifndef INCLUDE_BASIC_CUDA_MATRIX_H_
#define INCLUDE_BASIC_CUDA_MATRIX_H_

#include <automy/math/Matrix.h>
#include <automy/basic_cuda/CUDA.h>


namespace automy {
namespace basic_cuda {

template<typename T, int Rows, int Cols>
class CUDA_Matrix {
public:
	typedef T Scalar;
	
	CUDA_Matrix() {
		cuda_check(cudaMalloc(&data, Rows * Cols * sizeof(T)));
	}
	
	CUDA_Matrix(const CUDA_Matrix& mat) : CUDA_Matrix() {
		upload(mat);
	}
	
	template<typename S>
	CUDA_Matrix(const math::Matrix<S, Rows, Cols>& mat) : CUDA_Matrix() {
		upload(mat);
	}
	
	~CUDA_Matrix() {
		cuda_check(cudaFree(data));
	}
	
	size_t size() const {
		return Rows * Cols;
	}
	
	T* get_data() {
		return data;
	}
	
	void set_zero() {
		cuda_check(cudaMemset(data, 0, size() * sizeof(T)), "cudaMemset(): ");
	}
	
	void set_zero_async() {
		cuda_check(cudaMemsetAsync(data, 0, size() * sizeof(T)), "cudaMemsetAsync(): ");
	}
	
	CUDA_Matrix& operator=(const CUDA_Matrix& mat) {
		cuda_check(cudaMemcpy(data, mat.data, size() * sizeof(T), cudaMemcpyDeviceToDevice), "cudaMemcpy(): ");
		return *this;
	}
	
	template<typename S>
	CUDA_Matrix& operator=(const math::Matrix<S, Rows, Cols>& mat) {
		upload(mat);
		return *this;
	}
	
	template<typename S>
	void upload(const math::Matrix<S, Rows, Cols>& mat) {
		math::Matrix<T, Rows, Cols> tmp(mat);
		cuda_check(cudaMemcpy(data, tmp.get_data(), size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy(): ");
	}
	
	template<typename S>
	void upload_async(const math::Matrix<S, Rows, Cols>& mat, cudaStream_t stream = 0) {
		math::Matrix<T, Rows, Cols> tmp(mat);
		cuda_check(cudaMemcpyAsync(data, tmp.get_data(), size() * sizeof(T), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(): ");
	}
	
	math::Matrix<T, Rows, Cols> download() const {
		math::Matrix<T, Rows, Cols> res;
		cuda_check(cudaMemcpy(res.get_data(), data, size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy(): ");
		return res;
	}
	
private:
	T* data = 0;
	
};


} // basic_cuda
} // automy

#endif /* INCLUDE_BASIC_CUDA_MATRIX_H_ */
