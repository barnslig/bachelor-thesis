cmake_minimum_required(VERSION 3.10)

project(grapple LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-Wall -Wextra -pedantic -Werror)

add_executable(grapple main.cpp Grapple.cu)
