/*
 * CUDA_Texture.h
 *
 *  Created on: Feb 12, 2018
 *      Author: dev
 */

#ifndef INCLUDE_BASIC_CUDA_TEXTURE_H_
#define INCLUDE_BASIC_CUDA_TEXTURE_H_

#include <automy/basic_cuda/CUDA.h>
#include <automy/basic_cuda/CUDA_Image8.hpp>
#include <automy/basic/Image8.hpp>


namespace automy {
namespace basic_cuda {

class CUDA_Texture {
public:
	CUDA_Texture(int width, int height, int depth, int channels, int sizeof_channel)
		:	width(width), height(height), depth(depth), channels(channels), sizeof_channel(sizeof_channel),
			array(), res_desc(), tex_desc(), object()
	{
	}

	~CUDA_Texture() {
		cudaDestroyTextureObject(object);
		cudaFreeArray(array);
	}

	CUDA_Texture(const CUDA_Texture&) = delete;
	CUDA_Texture& operator=(const CUDA_Texture&) = delete;

	int stride() const {
		return width * channels * sizeof_channel;
	}

	void upload(const CUDA_Image8& image) {
		check_dimensions(image);
		cuda_check(cudaMemcpy2DToArray(array, 0, 0, image.data(), stride(), stride(), height, cudaMemcpyDeviceToDevice));
	}

	void upload_async(const CUDA_Image8& image, cudaStream_t stream = 0) {
		check_dimensions(image);
		cuda_check(cudaMemcpy2DToArrayAsync(array, 0, 0, image.data(), stride(), stride(), height, cudaMemcpyDeviceToDevice, stream));
	}

	void upload(const basic::Image8& image) {
		check_dimensions(image);
		cuda_check(cudaMemcpy2DToArray(array, 0, 0, image.data(), stride(), stride(), height, cudaMemcpyHostToDevice));
	}

	void upload_async(const basic::Image8& image, cudaStream_t stream = 0) {
		check_dimensions(image);
		cuda_check(cudaMemcpy2DToArrayAsync(array, 0, 0, image.data(), stride(), stride(), height, cudaMemcpyHostToDevice, stream));
	}

	CUDA_Texture& operator=(const CUDA_Image8& image) {
		upload(image);
		return *this;
	}

	CUDA_Texture& operator=(const basic::Image8& image) {
		upload(image);
		return *this;
	}

protected:
	void allocate(	cudaChannelFormatDesc channel_format,
					cudaTextureAddressMode address_mode,
					cudaTextureFilterMode filter_mode,
					cudaTextureReadMode read_mode,
					bool normalized_coords)
	{
		if(depth < 0) {
			cuda_check(cudaMallocArray(&array, &channel_format, width, height, cudaArrayTextureGather));
		} else {
			cudaExtent extend;
			extend.width = width;
			extend.height = height;
			extend.depth = depth;
			cuda_check(cudaMalloc3DArray(&array, &channel_format, extend, cudaArrayLayered | cudaArrayTextureGather));
		}
		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = array;

		tex_desc.addressMode[0] = address_mode;
		tex_desc.addressMode[1] = address_mode;
		tex_desc.addressMode[2] = address_mode;
		tex_desc.filterMode = filter_mode;
		tex_desc.normalizedCoords = normalized_coords ? 1 : 0;
		tex_desc.readMode = read_mode;

		cuda_check(cudaCreateTextureObject(&object, &res_desc, &tex_desc, 0));
	}
	
private:
	template<typename T>
	void check_dimensions(const T& image) const {
		if(int(image.width()) != width || int(image.height()) != height || int(image.depth()) != channels) {
			throw std::logic_error("image.width != width || image.height != height || image.depth != channels");
		}
	}

public:
	int width = 0;
	int height = 0;
	int depth = -1;
	int channels = 0;
	int sizeof_channel = 0;
	
	cudaArray_t array;
	cudaResourceDesc res_desc;
	cudaTextureDesc tex_desc;
	cudaTextureObject_t object;
	
};


class CUDA_TextureMono : public CUDA_Texture {
public:
	CUDA_TextureMono(	int width, int height, int depth = -1,
						cudaTextureAddressMode address_mode = cudaAddressModeClamp,
						cudaTextureFilterMode filter_mode = cudaFilterModeLinear,
						cudaTextureReadMode read_mode = cudaReadModeNormalizedFloat,
						bool normalized_coords = true)
		:	CUDA_Texture(width, height, depth, 1, 1)
	{
		cudaChannelFormatDesc channel_format = {};
		channel_format.f = cudaChannelFormatKindUnsigned;
		channel_format.x = 8;

		allocate(channel_format, address_mode, filter_mode, read_mode, normalized_coords);
	}

};


class CUDA_TextureRGBA : public CUDA_Texture {
public:
	CUDA_TextureRGBA(	int width, int height, int depth = -1,
						cudaTextureAddressMode address_mode = cudaAddressModeClamp,
						cudaTextureFilterMode filter_mode = cudaFilterModeLinear,
						cudaTextureReadMode read_mode = cudaReadModeNormalizedFloat,
						bool normalized_coords = true)
		:	CUDA_Texture(width, height, depth, 4, 1)
	{
		cudaChannelFormatDesc channel_format = {};
		channel_format.f = cudaChannelFormatKindUnsigned;
		channel_format.x = 8;
		channel_format.y = 8;
		channel_format.z = 8;
		channel_format.w = 8;
		
		allocate(channel_format, address_mode, filter_mode, read_mode, normalized_coords);
	}

};


} // basic_cuda
} // automy

#endif /* INCLUDE_BASIC_CUDA_TEXTURE_H_ */
