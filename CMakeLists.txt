cmake_minimum_required(VERSION 3.23)
project(CUDA_SHA_256 CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(CUDA_SHA_256 main.cu)

set_target_properties(CUDA_SHA_256 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)