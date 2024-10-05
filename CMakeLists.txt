cmake_minimum_required(VERSION 3.28)

project(cufft-server C)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER gcc-12)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(cufft_server SHARED
    src/cufft_server.cu
)
target_include_directories(cufft_server PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(cufft_server PUBLIC include)
set_target_properties(cufft_server PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cufft_server PRIVATE CUDA::cufft CUDA::cudart m)
