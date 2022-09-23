cmake_minimum_required(VERSION 3.22.0)

project(lbvh LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_CUDA_STANDARD 17)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()

set (CMAKE_CUDA_SEPARABLE_COMPILATION on)

add_executable(main 
lbvh.cuh
tiny_obj_loader.cpp
main.cu
)

target_include_directories(main
PUBLIC 
${CUDA_INCLUDE_DIRS})

target_link_libraries(main
${CUDA_LIBRARIES})
