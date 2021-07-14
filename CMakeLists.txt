cmake_minimum_required(VERSION 3.10)

project(grapple LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_SEPARABLE_COMPILATION True)

# makes nvcc mad
# add_compile_options(-Wall -Wextra -pedantic -Werror)

# add_executable(grapple main.cpp Grapple.cu)

add_executable(Queue Queue-main.cu Queue.cu)
