
#include <chrono>
#include <iostream>
#include <iterator>
#include <ratio>

#include "aabb.h"
#include "bvh.h"

#include "geometry.h"
#include "kernel.h"

#include "display/frame.h"
#include "display/gui.h"
#include "display/window.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

using tbvh =
    bvh<int3, float4, triangle_mesh_aabb, default_morton_code_calculator>;

using Clock = std::chrono::high_resolution_clock;
using namespace std;

void glfwWindowResize(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
    return 1;
  }

  display::WindowDescription desc;
  desc.title = "demo";
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
  display::Frame frame(800, 600);

  // save the content in glTexture to an image
  frame.save("test.png");

  std::function<void()> save = [&frame]() {
    frame.save("/home/yiwenxue/Pictures/cornell_dump.png");
  };

  std::function<void()> clear = [&frame]() { frame.clear(); };

  auto timeStemp = Clock::now();

  // load the mesh from disk
  std::vector<std::shared_ptr<Vertices>> vertices_pool;
  std::vector<TriangleMesh> meshes;

  auto loader = [&](std::string filename) {
    vertices_pool.emplace_back();
    auto &vertices = vertices_pool.back();
    auto instances = loadMesh(filename, vertices);
    std::move(instances.begin(), instances.end(), std::back_inserter(meshes));
  };

  loader(std::string(argv[1]));

  cout << "load mesh time: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                                timeStemp)
              .count()
       << "ms" << endl;

  timeStemp = Clock::now();
  for (auto &mesh : meshes) {
    cout << "triangles: " << mesh.indices.size() << endl;
    const auto &pos = mesh.m_mesh->pos;
    tbvh bvh(mesh.indices.begin(), mesh.indices.end(), pos.begin(), pos.end());
  }

  thrust::host_vector<int3> h_indices(100);
  thrust::host_vector<float4> vertex(100);

  cout << "bvh construct time: "
       << std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                                timeStemp)
              .count()
       << "us" << endl;

  while (!window.shouldClose()) {
    window.pollEvent();

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    frame.present();
    gui.begin();

    gui.window("frame", [&]() {
      auto ext = frame.getSize();
      gui.text("frame size: %dx%d", ext.x, ext.y);
      gui.button("clear", clear);
      gui.button("save", save);
    });

    gui.end();

    window.swapBuffers();
  }

  return 0;
}
