add_executable(frame
frame.cpp
kernel.cu
../bvh.cu
../camera.cpp
)

target_include_directories(frame
PUBLIC 
${CUDA_INCLUDE_DIRS})

target_link_libraries(frame
loader
display
${CUDA_LIBRARIES})


target_compile_options(frame
    PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;>
)
