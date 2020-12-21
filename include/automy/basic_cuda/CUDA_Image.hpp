/*
 * CUDA_Image.hpp
 *
 *  Created on: Feb 11, 2018
 *      Author: mad
 */

#ifndef INCLUDE_BASIC_CUDA_IMAGE_HPP_
#define INCLUDE_BASIC_CUDA_IMAGE_HPP_

#include <automy/basic_cuda/CUDA_Image.h>

#include <vnx/Input.h>
#include <vnx/Output.h>
#include <vnx/Visitor.h>
#include <vnx/DefaultPrinter.h>


namespace automy {
namespace basic_cuda {

template<typename T>
void CUDA_Image_read(vnx::TypeInput& in, CUDA_Image<T>& value, const vnx::TypeCode* type_code, const uint16_t* code) {
	vnx::skip(in, type_code, code);
	value.clear();
}

template<typename T>
void CUDA_Image_write(vnx::TypeOutput& out, const CUDA_Image<T>& value, const vnx::TypeCode* type_code, const uint16_t* code) {
	if(!type_code) {
		throw std::logic_error("write(): !type_code");		// only allowed inside another class
	}
	vnx::write_dynamic_null(out);
}

template<typename T>
void CUDA_Image_read(std::istream& in, CUDA_Image<T>& value) {
	// not supported
}

template<typename T>
void CUDA_Image_write(std::ostream& out, const CUDA_Image<T>& value) {
	vnx::DefaultPrinter printer(out);
	CUDA_Image_accept(printer, value);
}

template<typename T>
void CUDA_Image_accept(vnx::Visitor& visitor, const CUDA_Image<T>& value) {
	vnx::accept_image<T, 3>(visitor, 0, {value.width(), value.height(), value.depth()});
}


} // basic_cuda
} // automy

#endif /* INCLUDE_BASIC_CUDA_IMAGE_HPP_ */
