project (display)

find_package(OpenGL REQUIRED)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_library(display
    shader.cpp
    gui.cpp
    window.cpp
    frame.cpp
)

target_include_directories(display
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(display
PUBLIC
    glad
    imgui

PRIVATE
    glfw
    loader
    ${OPENGL_LIBRARIES}
    ${CUDA_LIBRARIES}
)
