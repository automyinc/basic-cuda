/*
 * CUDA_Image8.h
 *
 *  Created on: Jan 12, 2018
 *      Author: mad
 */

#ifndef INCLUDE_BASIC_CUDA_IMAGE8_H_
#define INCLUDE_BASIC_CUDA_IMAGE8_H_

#include <automy/basic_cuda/CUDA_Image.h>
#include <automy/basic/Image8.hpp>


namespace automy {
namespace basic_cuda {

class CUDA_Image8 : public CUDA_Image<uint8_t> {
public:
	CUDA_Image8() {}
	
	CUDA_Image8(size_t width_, size_t height_, size_t depth_ = 1) : CUDA_Image(width_, height_, depth_) {}
	
	CUDA_Image8(const CUDA_Image<uint8_t>& image) : CUDA_Image(image) {}
	
	CUDA_Image8& operator=(const basic::Image8& image) {
		*((CUDA_Image<uint8_t>*)this) = image;
		return *this;
	}
	
	static std::shared_ptr<CUDA_Image8> create() {
		return std::make_shared<CUDA_Image8>();
	}
	
};


} // basic_cuda
} // automy


namespace vnx {

void read(TypeInput& in, ::automy::basic_cuda::CUDA_Image8& value, const TypeCode* type_code, const uint16_t* code);

void write(TypeOutput& out, const ::automy::basic_cuda::CUDA_Image8& value, const TypeCode* type_code, const uint16_t* code);

void read(std::istream& in, ::automy::basic_cuda::CUDA_Image8& value);

void write(std::ostream& out, const ::automy::basic_cuda::CUDA_Image8& value);

void accept(Visitor& visitor, const ::automy::basic_cuda::CUDA_Image8& value);


} // vnx

#endif /* INCLUDE_BASIC_CUDA_IMAGE8_H_ */
