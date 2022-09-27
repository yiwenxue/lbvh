project(externals)

find_package(OpenGL REQUIRED)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_library(glad "glad/glad.c")
target_include_directories(glad 
PUBLIC 
    ./
PRIVATE 
    ./glad/)

add_library(imgui 
    imgui/imgui.h
    imgui/imgui.cpp
    imgui/imgui_draw.cpp 
    imgui/imgui_demo.cpp
    imgui/imgui_tables.cpp 
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp 
    imgui/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui 
PUBLIC 
    ./ 
    ./imgui ${GLFW_INCLUDE_DIR})

target_link_libraries(imgui
PRIVATE 
    ${OpenGL_LIBRARIES} ${GLFW_LIBRARY})

add_library(loader
    loader/stb_image.h
    loader/stb_image_write.h
    loader/tiny_gltf.h
    loader/tiny_obj_loader.h
    loader/json.hpp
    loader/loader.cpp
)
target_include_directories(loader 
PUBLIC
    ./
PRIVATE
    ./loader)
