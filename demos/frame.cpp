
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

#include "../bvh.h"
#include "../kernel.h"
#include "geometry.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

void glfwWindowResize(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

extern void renderer(float4 *framebuffer, int width, int height);

extern void renderSurf(cudaSurfaceObject_t surf, int width, int height);

extern void render_frame(cudaSurfaceObject_t surf, int width, int height);

extern void render_frame(cudaSurfaceObject_t surf, int width, int height,
                         const bvh_tree<int3, float4, true> &tree);

#include "../bvh.h"
#include "../camera.h"

template <typename Object, typename BufferType>
void rayTracer(const camera &cam, int2 frameSize,
               const bvh_tree<Object, BufferType, true> &root) {}

using tbvh =
    bvh<int3, float4, triangle_mesh_aabb, default_morton_code_calculator>;

int main(int argc, char **argv) {
  display::WindowDescription desc;
  desc.title = "demo";
  desc.position = {100, 100};
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

  std::vector<std::shared_ptr<Vertices>> vertices_pool;
  std::vector<TriangleMesh> meshes;

  auto loader = [&](std::string filename) {
    vertices_pool.emplace_back();
    auto &vertices = vertices_pool.back();
    auto instances = loadMesh(filename, vertices);
    std::move(instances.begin(), instances.end(), std::back_inserter(meshes));
  };

  loader("/home/yiwenxue/program/spatial/dragon.obj");

  // for first mesh, build bvh
  const auto &mesh = meshes[0];
  const auto &pos = mesh.m_mesh->pos;
  tbvh tree(mesh.indices.begin(), mesh.indices.end(), pos.begin(), pos.end());

  auto ctree = tree.get_ctree();

  display::GUI gui(&window);
  display::Frame frame(1280, 720);

  const auto extent = frame.getSize();
  const auto width = extent.x;
  const auto height = extent.y;

  auto cusurf = frame.getSurf();

  std::function<void()> render = [&]() {
    std::cout << "rendering" << std::endl;
    render_frame(cusurf, width, height, ctree);
    cudaDeviceSynchronize();
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

    // render_frame(cusurf, width, height);
    // render_frame(cusurf, width, height, ctree);
  }

  return 0;
}
