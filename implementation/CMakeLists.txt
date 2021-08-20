cmake_minimum_required(VERSION 3.10)

project(grapple LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_SEPARABLE_COMPILATION True)

# makes nvcc mad
# add_compile_options(-Wall -Wextra -pedantic -Werror)

add_executable(grapple src/main.cu src/Grapple.cu src/models/WaypointsState.cu src/models/PhilosophersState.cu)

## Unit Tests ##
include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(Hashtable_test src/Hashtable_test.cu)
target_link_libraries(Hashtable_test gtest_main)

add_executable(Queue_test src/Queue_test.cu)
target_link_libraries(Queue_test gtest_main)

add_executable(OutputBuffer_test src/OutputBuffer_test.cu)
target_link_libraries(OutputBuffer_test gtest_main)

add_executable(PhilosophersState_test src/models/PhilosophersState_test.cu src/models/PhilosophersState.cu)
target_link_libraries(PhilosophersState_test gtest_main)

add_executable(WaypointsState_test src/models/WaypointsState_test.cu src/models/WaypointsState.cu)
target_link_libraries(WaypointsState_test gtest_main)

include(GoogleTest)
gtest_discover_tests(Hashtable_test)
gtest_discover_tests(Queue_test)
gtest_discover_tests(OutputBuffer_test)
gtest_discover_tests(PhilosophersState_test)
gtest_discover_tests(WaypointsState_test)
