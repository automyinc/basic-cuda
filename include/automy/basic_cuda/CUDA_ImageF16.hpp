/*
 * CUDA_ImageF16.h
 *
 *  Created on: Dec 7, 2018
 *      Author: mad
 */

#ifndef INCLUDE_BASIC_CUDA_IMAGEF16_H_
#define INCLUDE_BASIC_CUDA_IMAGEF16_H_

#include <automy/basic_cuda/CUDA_Image.h>
#include <automy/basic/Image16.hpp>


namespace automy {
namespace basic_cuda {

class CUDA_ImageF16 : public CUDA_Image<uint16_t> {
public:
	CUDA_ImageF16() {}
	
	CUDA_ImageF16(size_t width_, size_t height_, size_t depth_ = 1) : CUDA_Image(width_, height_, depth_) {}
	
	CUDA_ImageF16(const CUDA_Image<uint16_t>& image) : CUDA_Image(image) {}
	
	CUDA_ImageF16& operator=(const basic::Image16& image) {
		*((CUDA_Image<uint16_t>*)this) = image;
		return *this;
	}
	
	static std::shared_ptr<CUDA_ImageF16> create() {
		return std::make_shared<CUDA_ImageF16>();
	}
	
};


} // basic_cuda
} // automy


namespace vnx {

void read(TypeInput& in, ::automy::basic_cuda::CUDA_ImageF16& value, const TypeCode* type_code, const uint16_t* code);

void write(TypeOutput& out, const ::automy::basic_cuda::CUDA_ImageF16& value, const TypeCode* type_code, const uint16_t* code);

void read(std::istream& in, ::automy::basic_cuda::CUDA_ImageF16& value);

void write(std::ostream& out, const ::automy::basic_cuda::CUDA_ImageF16& value);

void accept(Visitor& visitor, const ::automy::basic_cuda::CUDA_ImageF16& value);


} // vnx

#endif /* INCLUDE_BASIC_CUDA_IMAGEF16_H_ */
