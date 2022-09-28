
#include <chrono>
#include <iostream>
#include <iterator>
#include <ratio>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include "display/frame.h"
#include "display/gui.h"
#include "display/window.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

void glfwWindowResize(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

extern void renderer(float4 *framebuffer, int width, int height);

extern void renderSurf(cudaSurfaceObject_t surf, int width, int height);

#include "../bvh.h"
#include "../camera.h"

template <typename Object, typename BufferType>
void rayTracer(const camera &cam, int2 frameSize,
               const bvh_tree<Object, BufferType, true> &root) {}

int main(int argc, char **argv) {
  display::WindowDescription desc;
  desc.title = "demo";
  desc.size = {1280, 720};
  desc.visible = true;
  desc.borderless = false;
  desc.resizeable = true;
  desc.centered = false;
  desc.vsync = false;
  desc.fullscreen = false;

  display::Window window(desc);
  auto windowHandle = window.getWindowHandle();
  glfwSetWindowSizeCallback(windowHandle, glfwWindowResize);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  display::GUI gui(&window);
  display::Frame frame(1280, 720);

  const auto extent = frame.getSize();
  const auto width = extent.x;
  const auto height = extent.y;

  float4 *framebuffer;
  cudaMalloc(&framebuffer, width * height * sizeof(float4));
  auto cusurf = frame.getSurf();

  std::function<void()> render = [&]() {
    std::cout << "rendering" << std::endl;
    {
        // renderer(framebuffer, width, height);
        // frame.load(framebuffer, width, height);
    } {
      renderSurf(cusurf, width, height);
    }
  };

  std::function<void()> save = [&]() {
    std::cout << "saving" << std::endl;
    frame.save("test.png");
  };

  std::function<void()> clear = [&]() {
    std::cout << "clearing" << std::endl;
    frame.clear();
  };

  while (!window.shouldClose()) {
    window.pollEvent();

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    frame.present();
    gui.begin();

    gui.window("frame", [&]() {
      auto ext = frame.getSize();
      gui.text("frame size: %dx%d", ext.x, ext.y);
      gui.button("render", render);
      gui.button("clear", clear);
      gui.button("save", save);
    });

    gui.end();

    window.swapBuffers();
  }

  cudaFree(framebuffer);

  return 0;
}
