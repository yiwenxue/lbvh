#pragma once

#include "types.h"

#include <GLFW/glfw3.h>
#include <string>

namespace display {

struct WindowDescription {
    std::string title;
    Offset2D    position   = {100, 100};
    Extent2D    size       = {640, 640};
    bool        visible    = false;
    bool        borderless = false;
    bool        resizeable = false;
    bool        centered   = false;
    bool        vsync      = false;
    bool        fullscreen = false;
};

class Window {
public:
    using WindowDesc   = WindowDescription;
    using WindowHandle = GLFWwindow *;

    Window(const WindowDescription &description);
    Window(const Window &)            = delete;
    Window(Window &&)                 = delete;
    Window &operator=(const Window &) = delete;
    Window &operator=(Window &&)      = delete;
    ~Window();

    void initialize(const WindowDescription &description);

    void setPosition(const Offset2D &position);
    void setSize(const Extent2D &size);
    void setTitle(const std::string &title);

    Offset2D     getPosition() const;
    Extent2D     getSize() const;
    std::string  getTitle() const;
    WindowDesc   getDescription() const;
    WindowHandle getWindowHandle() const;

    bool shouldClose() const;
    void show(bool show);
    bool isShown() const;
    void pollEvent() const;
    void swapBuffers() const;

protected:
    WindowDescription _desc{};
    bool              _quit    = false;
    bool              _focused = false;
    WindowHandle      _handle{nullptr};

    static bool          isGlfwInited;
    static std::uint32_t glfwWindowCount;
};

inline Window::WindowHandle Window::getWindowHandle() const { return _handle; }

inline Window::WindowDesc Window::getDescription() const { return _desc; }
} // namespace display
