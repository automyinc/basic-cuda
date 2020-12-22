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
	
	CUDA_Matrix() : CUDA_Matrix(1) {}

	CUDA_Matrix(size_t depth_)
		:	depth_(depth_)
	{
		cuda_check(cudaMalloc(&data_, size() * sizeof(T)));
	}
	
	CUDA_Matrix(const CUDA_Matrix& mat) : CUDA_Matrix() {
		upload(mat);
	}
	
	template<typename S>
	CUDA_Matrix(const math::Matrix<S, Rows, Cols>& mat) : CUDA_Matrix() {
		upload(mat);
	}
	
	~CUDA_Matrix() {
		cuda_check(cudaFree(data_));
	}
	
	size_t size() const {
		return Rows * Cols * depth_;
	}
	
	size_t depth() const {
		return depth_;
	}

	T* data() {
		return data_;
	}
	
	const T* data() const {
		return data_;
	}

	void set_zero() {
		cuda_check(cudaMemset(data_, 0, size() * sizeof(T)), "cudaMemset(): ");
	}
	
	void set_zero_async(cudaStream_t stream = 0) {
		cuda_check(cudaMemsetAsync(data_, 0, size() * sizeof(T), stream), "cudaMemsetAsync(): ");
	}
	
	CUDA_Matrix& operator=(const CUDA_Matrix& mat) {
		if(mat.size() != size()) {
			throw std::logic_error("size mismatch");
		}
		cuda_check(cudaMemcpy(data_, mat.data_, size() * sizeof(T), cudaMemcpyDeviceToDevice), "cudaMemcpy(): ");
		return *this;
	}
	
	template<typename S>
	CUDA_Matrix& operator=(const math::Matrix<S, Rows, Cols>& mat) {
		upload(mat);
		return *this;
	}
	
	template<typename S>
	void upload(const math::Matrix<S, Rows, Cols>& mat) {
		if(depth_ != 1) {
			throw std::logic_error("depth != 1");
		}
		const math::Matrix<T, Rows, Cols> tmp(mat);
		cuda_check(cudaMemcpy(data_, tmp.get_data(), size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy(): ");
	}
	
	template<typename S>
	void upload_async(const math::Matrix<S, Rows, Cols>& mat, cudaStream_t stream = 0) {
		if(depth_ != 1) {
			throw std::logic_error("depth != 1");
		}
		const math::Matrix<T, Rows, Cols> tmp(mat);
		cuda_check(cudaMemcpyAsync(data_, tmp.get_data(), size() * sizeof(T), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(): ");
	}
	
	template<typename S>
	void upload_async(const std::vector<math::Matrix<S, Rows, Cols>>& mats, cudaStream_t stream = 0) {
		if(mats.size() != depth_) {
			throw std::logic_error("size mismatch");
		}
		const std::vector<math::Matrix<T, Rows, Cols>> tmp(mats.begin(), mats.end());
		cuda_check(cudaMemcpyAsync(data_, tmp[0].get_data(), size() * sizeof(T), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(): ");
	}

	math::Matrix<T, Rows, Cols> download() const {
		if(depth_ != 1) {
			throw std::logic_error("depth != 1");
		}
		math::Matrix<T, Rows, Cols> res;
		cuda_check(cudaMemcpy(res.get_data(), data_, size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy(): ");
		return res;
	}
	
	std::vector<math::Matrix<T, Rows, Cols>> download_all() const {
		std::vector<math::Matrix<T, Rows, Cols>> res;
		res.resize(depth_);
		cuda_check(cudaMemcpy(res[0].get_data(), data_, size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy(): ");
		return res;
	}

private:
	T* data_ = 0;
	size_t depth_ = 1;
	
};


} // basic_cuda
} // automy

#endif /* INCLUDE_BASIC_CUDA_MATRIX_H_ */
