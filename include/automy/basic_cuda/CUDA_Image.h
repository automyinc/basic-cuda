/*
 * CUDA_Image.h
 *
 *  Created on: Jan 12, 2018
 *      Author: mad
 */

#ifndef INCLUDE_BASIC_CUDA_IMAGE_H_
#define INCLUDE_BASIC_CUDA_IMAGE_H_

#include <automy/basic/Image.h>
#include <automy/basic_cuda/CUDA.h>


namespace automy {
namespace basic_cuda {

template<typename T>
class CUDA_Image {
public:
	CUDA_Image() {}
	
	CUDA_Image(size_t width_, size_t height_, size_t depth_ = 1) {
		resize(width_, height_, depth_);
	}
	
	CUDA_Image(const CUDA_Image& image) {
		*this = image;
	}
	
	CUDA_Image(const basic::Image<T>& image) {
		*this = image;
	}
	
	~CUDA_Image() {
		if(data) {
			cudaFree(data);
		}
		data = 0;
	}
	
	CUDA_Image& operator=(const CUDA_Image& image) {
		resize(image.width(), image.height(), image.depth());
		cuda_check(cudaMemcpy(data, image.get_data(), get_size() * sizeof(T), cudaMemcpyDeviceToDevice), "cudaMemcpy() :");
		return *this;
	}
	
	CUDA_Image& operator=(const basic::Image<T>& image) {
		upload(image);
		return *this;
	}
	
	uint32_t width() const {
		return width_;
	}
	
	uint32_t height() const {
		return height_;
	}
	
	uint32_t depth() const {
		return depth_;
	}
	
	size_t get_size() const {
		return size_t(width_) * size_t(height_) * size_t(depth_);
	}
	
	T* get_data() {
		return data;
	}
	
	const T* get_data() const {
		return data;
	}
	
	void resize(size_t new_width, size_t new_height, size_t new_depth = 1) {
		const size_t new_size = new_width * new_height * new_depth;
		if(new_size != get_size()) {
			if(data) {
				cudaFree(data);
			}
			cuda_check(cudaMalloc((void**)&data, new_size * sizeof(T)), "cudaMallocManaged(): ");
		}
		width_ = new_width;
		height_ = new_height;
		depth_ = new_depth;
	}
	
	void clear() {
		width_ = 0;
		height_ = 0;
		depth_ = 0;
		if(data) {
			cudaFree(data);
		}
		data = 0;
	}
	
	void set_zero() {
		cuda_check(cudaMemset(data, 0, get_size() * sizeof(T)), "cudaMemset(): ");
	}
	
	void set_zero_async() {
		cuda_check(cudaMemsetAsync(data, 0, get_size() * sizeof(T)), "cudaMemsetAsync(): ");
	}
	
	void copy_from_async(const CUDA_Image& image) {
		resize(image.width(), image.height(), image.depth());
		cuda_check(cudaMemcpyAsync(data, image.get_data(), get_size() * sizeof(T), cudaMemcpyDeviceToDevice), "cudaMemcpyAsync() :");
	}
	
	void upload(const basic::Image<T>& image) {
		resize(image.width(), image.height(), image.depth());
		cuda_check(cudaMemcpy(data, image.get_data(), get_size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy() :");
	}
	
	void upload_async(const basic::Image<T>& image) {
		resize(image.width(), image.height(), image.depth());
		cuda_check(cudaMemcpyAsync(data, image.get_data(), get_size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpyAsync() :");
	}
	
	void download(basic::Image<T>& image) const {
		image.resize(width_, height_, depth_);
		cuda_check(cudaMemcpy(image.get_data(), data, get_size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy() :");
	}
	
	void prefetch(int device) const {
		cuda_check(cudaMemPrefetchAsync(data, get_size() * sizeof(T), device), "cudaMemPrefetchAsync(): ");
	}
	
private:
	T* data = 0;
	uint32_t width_ = 0;
	uint32_t height_ = 0;
	uint32_t depth_ = 0;
	
};


template<typename T>
void cuda_host_register(basic::Image<T>& image) {
	cuda_check(cudaHostRegister(image.get_data(), image.get_size() * sizeof(T), cudaHostRegisterDefault));
}

template<typename T>
void cuda_host_unregister(basic::Image<T>& image) {
	cuda_check(cudaHostUnregister(image.get_data()));
}


} // basic_cuda
} // automy

#endif /* INCLUDE_BASIC_CUDA_IMAGE_H_ */
