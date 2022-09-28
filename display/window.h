#pragma once

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <string>

namespace display {

struct WindowDescription {
  std::string title;
  int2 position = make_int2(0, 0);
  int2 size = make_int2(640, 480);
  bool visible = false;
  bool borderless = false;
  bool resizeable = false;
  bool centered = false;
  bool vsync = false;
  bool fullscreen = false;
};

class Window {
public:
  using WindowDesc = WindowDescription;
  using WindowHandle = GLFWwindow *;

  Window(const WindowDescription &description);
  Window(const Window &) = delete;
  Window(Window &&) = delete;
  Window &operator=(const Window &) = delete;
  Window &operator=(Window &&) = delete;
  ~Window();

  void initialize(const WindowDescription &description);

  void setPosition(const int2 &position);
  void setSize(const int2 &size);
  void setTitle(const std::string &title);

  int2 getPosition() const;
  int2 getSize() const;
  std::string getTitle() const;
  WindowDesc getDescription() const;
  WindowHandle getWindowHandle() const;

  bool shouldClose() const;
  void show(bool show);
  bool isShown() const;
  void pollEvent() const;
  void swapBuffers() const;

protected:
  WindowDescription _desc{};
  bool _quit = false;
  bool _focused = false;
  WindowHandle _handle{nullptr};

  static bool isGlfwInited;
  static std::uint32_t glfwWindowCount;
};

inline Window::WindowHandle Window::getWindowHandle() const { return _handle; }

inline Window::WindowDesc Window::getDescription() const { return _desc; }
} // namespace display
