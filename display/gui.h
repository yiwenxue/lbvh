#pragma once

#include "types.h"
#include "window.h"

#include "imgui/imgui.h"

#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include <functional>

namespace display {

class GUI {
public:
    GUI(Window *window);
    ~GUI();

    void begin();
    void end();

    void window(const std::string &title, const Extent2D &size,
                const std::function<void()> &callback);

    void button(const std::string &title, const std::function<void()> &callback);

    void checkbox(const std::string &title, bool &value);

    void slider(const std::string &title, float &value, float min, float max);

    void slider(const std::string &title, int &value, int min, int max);

    void text(const std::string &title);

    void text(const char *fmt, ...);

    void text(const std::string &title, const std::string &value);

    void graph(const std::string &title, const std::vector<float> &values,
               float min, float max);

    void graph(const std::string &title, const std::vector<float> &values);

private:
    Window       *m_window;
    ImGuiContext *m_context;
    ImGuiIO      *m_io;
};

} // namespace display
