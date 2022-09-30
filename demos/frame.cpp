
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

// camera control
#include "../camera.h"

int2 frameSize = {1280, 720};
float currentFrame = 0;
float deltaTime    = 0;

Camera globalCamera;

double lastX = 0;
double lastY = 0;

bool mouseEnabled = false;

void glfwWindowResize(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

void glfwScrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    globalCamera.changeRadius(yoffset * -0.01);
    std::cout << "Radius: " << globalCamera.radius << std::endl;
}

void glfwMouseCallback(GLFWwindow *window, double xpos, double ypos)
{
    double xoffset = xpos - lastX;
    double yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    globalCamera.changePitch(yoffset * -0.01 * mouseEnabled);
    globalCamera.changeYaw(xoffset * -0.01 * mouseEnabled);
}

void glfwMouseKeyCallback(GLFWwindow *window, int key, int action, int mods)
{
    if (key == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && !mouseEnabled)
    {
        glfwGetCursorPos(window, &lastX, &lastY);
        mouseEnabled = true;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
}

void glfwKeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS)
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        if (mouseEnabled)
        {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            mouseEnabled = false;
        }
        else
            glfwSetWindowShouldClose(window, true);
}

void processInput(GLFWwindow *window, float delta)
{
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        globalCamera.move(10 * delta);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        globalCamera.move(-10 * delta);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        globalCamera.strafe(-10 * delta);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        globalCamera.strafe(10 * delta);

    // std::cout << "center: " << globalCamera.centerPosition.x << ", " << globalCamera.centerPosition.y << ", " << globalCamera.centerPosition.z << std::endl;
}

void glfwErrorCallback(int error, const char *description)
{
    std::cerr << "GLFW error: " << description << std::endl;
}

extern void renderer(float4 *framebuffer, int width, int height);

extern void renderSurf(cudaSurfaceObject_t surf, int width, int height);

extern void render_frame(cudaSurfaceObject_t surf, int width, int height, camera_data &cam);

extern void render_frame(cudaSurfaceObject_t surf, int width, int height, camera_data &cam,
                         const bvh_tree<int3, float4, true> &tree);

#include "../bvh.h"
#include "../camera.h"

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
  glfwSetKeyCallback(windowHandle, glfwKeyCallback);
  glfwSetErrorCallback(glfwErrorCallback);
  glfwSetWindowSizeCallback(windowHandle, glfwWindowResize);
  glfwSetFramebufferSizeCallback(windowHandle, glfwWindowResize);
  glfwSetCursorPosCallback(windowHandle, glfwMouseCallback);
  glfwSetScrollCallback(windowHandle, glfwScrollCallback);
  glfwSetMouseButtonCallback(windowHandle, glfwMouseKeyCallback);

  globalCamera.setRes(frameSize.x, frameSize.y);
  camera_data cam_data;

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

  loader("./dragon.obj");
  // loader("./000.obj");

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

  std::function<void()> visualize = [&]() {
    bvh_tree_visualizer(ctree);
    std::cout << "save to dot" << std::endl;
  };

  std::function<void()> render = [&]() {
    globalCamera.buildRenderCamera(cam_data);
    // Timer timer("bvh render");
    render_frame(cusurf, width, height, cam_data, ctree);
  };

  std::function<void()> save = [&]() {
    std::cout << "saving" << std::endl;
    frame.save("test.png");
  };

  std::function<void()> clear = [&]() {
    std::cout << "clearing" << std::endl;
    frame.clear();
  };
  static float lastFrame = 0;

  while (!window.shouldClose()) {
    window.pollEvent();
    currentFrame = static_cast<float>(glfwGetTime());
    deltaTime    = currentFrame - lastFrame;
    lastFrame    = currentFrame;

    processInput(windowHandle, deltaTime);

    {
      globalCamera.buildRenderCamera(cam_data);
      // Timer timer("sphere render");
      render_frame(cusurf, width, height, cam_data);
    }
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    frame.present();
    gui.begin();

    gui.window("frame", [&]() {
      auto ext = frame.getSize();
      gui.text("frame size: %dx%d", ext.x, ext.y);
      gui.button("visualize", visualize);
      gui.button("render", render);
      gui.button("clear", clear);
      gui.button("save", save);
    });

    gui.end();

    window.swapBuffers();
  }

  return 0;
}
