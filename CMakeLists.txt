cmake_minimum_required(VERSION 3.28)

project(cufft-server C CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER g++-12)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(test_add
    src/add.cu
)
#target_compile_options(test_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#    --fast-math
#)
target_include_directories(test_add PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(test_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_add PRIVATE CUDA::cufft CUDA::cudart m)
