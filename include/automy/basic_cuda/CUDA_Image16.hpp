/*
 * CUDA_Image16.h
 *
 *  Created on: Jan 12, 2018
 *      Author: mad
 */

#ifndef INCLUDE_BASIC_CUDA_IMAGE16_H_
#define INCLUDE_BASIC_CUDA_IMAGE16_H_

#include <automy/basic_cuda/CUDA_Image.h>
#include <automy/basic/Image16.hpp>


namespace automy {
namespace basic_cuda {

class CUDA_Image16 : public CUDA_Image<uint16_t> {
public:
	CUDA_Image16() {}
	
	CUDA_Image16(size_t width_, size_t height_, size_t depth_ = 1) : CUDA_Image(width_, height_, depth_) {}
	
	CUDA_Image16(const CUDA_Image<uint16_t>& image) : CUDA_Image(image) {}
	
	CUDA_Image16& operator=(const basic::Image16& image) {
		*((CUDA_Image<uint16_t>*)this) = image;
		return *this;
	}
	
	static std::shared_ptr<CUDA_Image16> create() {
		return std::make_shared<CUDA_Image16>();
	}
	
};


} // basic_cuda
} // automy


namespace vnx {

void read(TypeInput& in, ::automy::basic_cuda::CUDA_Image16& value, const TypeCode* type_code, const uint16_t* code);

void write(TypeOutput& out, const ::automy::basic_cuda::CUDA_Image16& value, const TypeCode* type_code, const uint16_t* code);

void read(std::istream& in, ::automy::basic_cuda::CUDA_Image16& value);

void write(std::ostream& out, const ::automy::basic_cuda::CUDA_Image16& value);

void accept(Visitor& visitor, const ::automy::basic_cuda::CUDA_Image16& value);


} // vnx

#endif /* INCLUDE_BASIC_CUDA_IMAGE16_H_ */
