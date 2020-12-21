/*
 * CUDA_ImageF32.cpp
 *
 *  Created on: Feb 7, 2018
 *      Author: dev
 */

#include <automy/basic_cuda/CUDA_ImageF32.hpp>
#include <automy/basic_cuda/CUDA_Image.hpp>


namespace vnx {

void read(TypeInput& in, ::automy::basic_cuda::CUDA_ImageF32& value, const TypeCode* type_code, const uint16_t* code) {
	CUDA_Image_read(in, value, type_code, code);
}

void write(TypeOutput& out, const ::automy::basic_cuda::CUDA_ImageF32& value, const TypeCode* type_code, const uint16_t* code) {
	CUDA_Image_write(out, value, type_code, code);
}

void read(std::istream& in, ::automy::basic_cuda::CUDA_ImageF32& value) {
	CUDA_Image_read(in, value);
}

void write(std::ostream& out, const ::automy::basic_cuda::CUDA_ImageF32& value) {
	CUDA_Image_write(out, value);
}

void accept(Visitor& visitor, const ::automy::basic_cuda::CUDA_ImageF32& value) {
	CUDA_Image_accept(visitor, value);
}


} // vnx
