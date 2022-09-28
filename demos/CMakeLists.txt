add_executable(frame
frame.cpp
kernel.cu
)

target_include_directories(frame
PUBLIC 
${CUDA_INCLUDE_DIRS})

target_link_libraries(frame
loader
display
${CUDA_LIBRARIES})
