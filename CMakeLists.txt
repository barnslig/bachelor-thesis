cmake_minimum_required(VERSION 3.10)

project(grapple LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_SEPARABLE_COMPILATION True)

# makes nvcc mad
# add_compile_options(-Wall -Wextra -pedantic -Werror)

add_executable(grapple main.cpp Grapple.cu Hashtable.cu Queue.cu State.cu)

## Unit Tests ##
include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(Hashtable_test Hashtable_test.cu Hashtable.cu)
target_link_libraries(Hashtable_test gtest_main)

add_executable(Queue_test Queue_test.cu Queue.cu)
target_link_libraries(Queue_test gtest_main)

include(GoogleTest)
gtest_discover_tests(Hashtable_test)
gtest_discover_tests(Queue_test)
