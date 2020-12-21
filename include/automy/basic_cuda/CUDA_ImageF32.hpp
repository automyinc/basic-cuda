/*
 * CUDA_ImageF32.h
 *
 *  Created on: Feb 7, 2018
 *      Author: mad
 */

#ifndef INCLUDE_BASIC_CUDA_IMAGEF32_H_
#define INCLUDE_BASIC_CUDA_IMAGEF32_H_

#include <automy/basic_cuda/CUDA_Image.h>
#include <automy/basic/ImageF32.hpp>


namespace automy {
namespace basic_cuda {

class CUDA_ImageF32 : public CUDA_Image<float> {
public:
	CUDA_ImageF32() {}
	
	CUDA_ImageF32(size_t width_, size_t height_, size_t depth_ = 1) : CUDA_Image(width_, height_, depth_) {}
	
	CUDA_ImageF32(const CUDA_Image<float>& image) : CUDA_Image(image) {}
	
	CUDA_ImageF32& operator=(const basic::ImageF32& image) {
		*((CUDA_Image<float>*)this) = image;
		return *this;
	}
	
	static std::shared_ptr<CUDA_ImageF32> create() {
		return std::make_shared<CUDA_ImageF32>();
	}
	
};


} // basic_cuda
} // automy


namespace vnx {

void read(TypeInput& in, ::automy::basic_cuda::CUDA_ImageF32& value, const TypeCode* type_code, const uint16_t* code);

void write(TypeOutput& out, const ::automy::basic_cuda::CUDA_ImageF32& value, const TypeCode* type_code, const uint16_t* code);

void read(std::istream& in, ::automy::basic_cuda::CUDA_ImageF32& value);

void write(std::ostream& out, const ::automy::basic_cuda::CUDA_ImageF32& value);

void accept(Visitor& visitor, const ::automy::basic_cuda::CUDA_ImageF32& value);


} // vnx

#endif /* INCLUDE_BASIC_CUDA_IMAGEF32_H_ */
