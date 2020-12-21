cmake_minimum_required(VERSION 3.2)
project(automy-basic-cuda)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} $ENV{CUDA_NVCC_FLAGS} -lineinfo --default-stream per-thread --disable-warnings")

file(GLOB GENERATED_SOURCE_FILES generated/src/*.cpp)

include_directories(
	BEFORE
	include
	generated/include
)

add_library(automy_basic_cuda SHARED
	${GENERATED_SOURCE_FILES}
	src/CUDA_Image8.cpp
	src/CUDA_Image16.cpp
	src/CUDA_ImageF16.cpp
	src/CUDA_ImageF32.cpp
	src/CUDA_Bridge.cpp
)

target_include_directories(automy_basic_cuda
	PUBLIC include generated/include ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(automy_basic_cuda
	automy_basic automy_math vnx_base cuda ${CUDA_LIBRARIES}
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY generated/include/ DESTINATION include)
install(DIRECTORY interface/ DESTINATION interface/automy/basic_cuda)

install(TARGETS automy_basic_cuda DESTINATION lib)
