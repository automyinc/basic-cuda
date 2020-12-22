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
		clear();
	}
	
	CUDA_Image& operator=(const CUDA_Image& image) {
		resize(image.width(), image.height(), image.depth());
		cuda_check(cudaMemcpy(data_, image.data(), size() * sizeof(T), cudaMemcpyDeviceToDevice), "cudaMemcpy() :");
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
	
	size_t size() const {
		return size_t(width_) * size_t(height_) * size_t(depth_);
	}
	
	size_t get_size() const {
		return size();
	}

	T* data() {
		return data_;
	}

	const T* data() const {
		return data_;
	}
	
	void resize(size_t new_width, size_t new_height, size_t new_depth = 1) {
		const size_t new_size = new_width * new_height * new_depth;
		if(new_size != size()) {
			if(data_) {
				cudaFree(data_);
				data_ = nullptr;
			}
			if(new_size) {
				cuda_check(cudaMalloc((void**)&data_, new_size * sizeof(T)), "cudaMallocManaged(): ");
			}
		}
		width_ = new_width;
		height_ = new_height;
		depth_ = new_depth;
	}
	
	void clear() {
		width_ = 0;
		height_ = 0;
		depth_ = 0;
		if(data_) {
			cudaFree(data_);
		}
		data_ = nullptr;
	}
	
	void set_zero() {
		cuda_check(cudaMemset(data_, 0, size() * sizeof(T)), "cudaMemset(): ");
	}
	
	void set_zero_async(cudaStream_t stream = 0) {
		cuda_check(cudaMemsetAsync(data_, 0, size() * sizeof(T), stream), "cudaMemsetAsync(): ");
	}
	
	void copy_from_async(const CUDA_Image& image, cudaStream_t stream = 0) {
		resize(image.width(), image.height(), image.depth());
		cuda_check(cudaMemcpyAsync(data_, image.data(), size() * sizeof(T), cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync() :");
	}
	
	void upload(const basic::Image<T>& image) {
		resize(image.width(), image.height(), image.depth());
		cuda_check(cudaMemcpy(data_, image.get_data(), size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy() :");
	}
	
	void upload_async(const basic::Image<T>& image, cudaStream_t stream = 0) {
		resize(image.width(), image.height(), image.depth());
		cuda_check(cudaMemcpyAsync(data_, image.get_data(), size() * sizeof(T), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync() :");
	}
	
	void download(basic::Image<T>& image) const {
		image.resize(width_, height_, depth_);
		cuda_check(cudaMemcpy(image.get_data(), data_, size() * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy() :");
	}
	
	void prefetch(int device, cudaStream_t stream = 0) const {
		cuda_check(cudaMemPrefetchAsync(data_, size() * sizeof(T), device, stream), "cudaMemPrefetchAsync(): ");
	}
	
private:
	T* data_ = nullptr;
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
