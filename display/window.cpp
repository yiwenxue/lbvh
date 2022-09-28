#include "window.h"
#include <cassert>

namespace display {

bool Window::isGlfwInited = false;
std::uint32_t Window::glfwWindowCount = 0;

Window::Window(const WindowDesc &description) { initialize(description); }

void Window::initialize(const WindowDesc &description) {
  _desc = description;

  if (!isGlfwInited) {
    glfwInit();
  }

  glfwWindowHint(GLFW_DEPTH_BITS, 32);
  glfwWindowHint(GLFW_STENCIL_BITS, 8);

  glfwWindowHint(GLFW_RED_BITS, 8);
  glfwWindowHint(GLFW_GREEN_BITS, 8);
  glfwWindowHint(GLFW_BLUE_BITS, 8);
  glfwWindowHint(GLFW_ALPHA_BITS, 8);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

  if (description.resizeable) {
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  } else {
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  }

  // Create different type of window fullscreen or not.
  if (description.fullscreen) {
    _handle = glfwCreateWindow(static_cast<int>(description.size.x),
                               static_cast<int>(description.size.y),
                               description.title.c_str(),
                               glfwGetPrimaryMonitor(), nullptr);
  } else {
    _handle = glfwCreateWindow(static_cast<int>(description.size.x),
                               static_cast<int>(description.size.y),
                               description.title.c_str(), nullptr, nullptr);
  }

  assert(_handle != nullptr && "GlfwWindow count be created properly\n");

  show(description.visible);

  glfwMakeContextCurrent(_handle);

  setPosition(description.position);

  glfwWindowCount++;
}

Window::~Window() {
  assert(_handle != nullptr && "GlfwWindow is destroied before desctruction\n");
  glfwDestroyWindow(_handle);
  glfwWindowCount--;

  if (glfwWindowCount == 0) {
    glfwTerminate();
    isGlfwInited = false;
  }
}

void Window::setPosition(const int2 &position) {
  glfwSetWindowPos(_handle, static_cast<int>(position.x),
                   static_cast<int>(position.y));
}

int2 Window::getPosition() const {
  int x;
  int y;
  glfwGetWindowPos(_handle, &x, &y);
  return make_int2(x, y);
}

void Window::setSize(const int2 &size) {
  glfwSetWindowSize(_handle, static_cast<int>(size.x),
                    static_cast<int>(size.y));
}

int2 Window::getSize() const {
  int x;
  int y;
  glfwGetWindowSize(_handle, &x, &y);
  return make_int2(x, y);
};

void Window::setTitle(const std::string &title) {
  glfwSetWindowTitle(_handle, title.c_str());
  _desc.title = title;
}

std::string Window::getTitle() const { return _desc.title; };

bool Window::shouldClose() const { return glfwWindowShouldClose(_handle); };

void Window::show(bool show) {
  if (show) {
    glfwShowWindow(_handle);
  } else {
    glfwHideWindow(_handle);
  }
};

bool Window::isShown() const {
  return glfwGetWindowAttrib(_handle, GLFW_VISIBLE);
}

void Window::pollEvent() const { glfwPollEvents(); }

void Window::swapBuffers() const { glfwSwapBuffers(_handle); }

} // namespace display
