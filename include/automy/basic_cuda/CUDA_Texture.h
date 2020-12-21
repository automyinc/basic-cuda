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


namespace automy {
namespace basic_cuda {

class CUDA_Texture {
public:
	CUDA_Texture(int width, int height)
		:	width(width), height(height), array(), res_desc(), tex_desc(), object()
	{
	}
	
	~CUDA_Texture() {
		cudaDestroyTextureObject(object);
		cudaFreeArray(array);
	}
	
	CUDA_Texture(const CUDA_Texture&) = delete;
	CUDA_Texture& operator=(const CUDA_Texture&) = delete;
	
	int width = 0;
	int height = 0;
	
	cudaArray_t array;
	cudaResourceDesc res_desc;
	cudaTextureDesc tex_desc;
	cudaTextureObject_t object;
	
};


class CUDA_TextureRGBA : public CUDA_Texture {
public:
	CUDA_TextureRGBA(	int width, int height, int depth = -1,
						cudaTextureAddressMode address_mode = cudaAddressModeClamp,
						cudaTextureFilterMode filter_mode = cudaFilterModeLinear,
						cudaTextureReadMode read_mode = cudaReadModeNormalizedFloat,
						bool normalized_coords = true)
		:	CUDA_Texture(width, height)
	{
		cudaChannelFormatDesc channel_format = {};
		channel_format.f = cudaChannelFormatKindUnsigned;
		channel_format.x = 8;
		channel_format.y = 8;
		channel_format.z = 8;
		channel_format.w = 8;
		
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
	
	void upload(const CUDA_Image8& image) {
		if(image.width() != width || image.height() != height || image.depth() != 4) {
			throw std::logic_error("image.width != width || image.height != height || image.depth != 4");
		}
		cuda_check(cudaMemcpy2DToArray(array, 0, 0, image.get_data(), width * 4, width * 4, height, cudaMemcpyDeviceToDevice));
	}
	
	void upload_async(const CUDA_Image8& image) {
		if(image.width() != width || image.height() != height || image.depth() != 4) {
			throw std::logic_error("image.width != width || image.height != height || image.depth != 4");
		}
		cuda_check(cudaMemcpy2DToArrayAsync(array, 0, 0, image.get_data(), width * 4, width * 4, height, cudaMemcpyDeviceToDevice));
	}
	
	CUDA_TextureRGBA& operator=(const CUDA_Image8& image) {
		upload(image);
		return *this;
	}
	
};


} // basic_cuda
} // automy

#endif /* INCLUDE_BASIC_CUDA_TEXTURE_H_ */
