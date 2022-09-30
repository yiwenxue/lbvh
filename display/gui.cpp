#include "gui.h"

namespace display {

GUI::GUI(Window *window) : m_window(window) {
  IMGUI_CHECKVERSION();
  m_context = ImGui::CreateContext();
  m_io = &ImGui::GetIO();
  m_io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(m_window->getWindowHandle(), true);
  ImGui_ImplOpenGL3_Init("#version 330 core");
}

GUI::~GUI() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext(m_context);
  m_context = nullptr;
  m_io = nullptr;
}

void GUI::begin() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

void GUI::end() {
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GUI::window(const std::string &title,
                 const std::function<void()> &callback) {
  ImGui::Begin(title.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  callback();
  ImGui::End();
}

void GUI::button(const std::string &title,
                 const std::function<void()> &callback) {
  if (ImGui::Button(title.c_str())) {
    callback();
  }
}

void GUI::checkbox(const std::string &title, bool &value) {
  ImGui::Checkbox(title.c_str(), &value);
}

void GUI::slider(const std::string &title, float &value, float min, float max) {
  ImGui::SliderFloat(title.c_str(), &value, min, max);
}

void GUI::slider(const std::string &title, int &value, int min, int max) {
  ImGui::SliderInt(title.c_str(), &value, min, max);
}

void GUI::text(const std::string &title) { ImGui::Text("%s", title.c_str()); }

void GUI::text(const std::string &title, const std::string &value) {
  ImGui::Text("%s: %s", title.c_str(), value.c_str());
}

void GUI::text(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  ImGui::TextV(fmt, args);
  va_end(args);
}

void GUI::graph(const std::string &title, const std::vector<float> &values,
                float min, float max) {
  ImGui::PlotLines(title.c_str(), values.data(), values.size(), 0, nullptr, min,
                   max);
}

void GUI::graph(const std::string &title, const std::vector<float> &values) {
  ImGui::PlotLines(title.c_str(), values.data(), values.size());
}

} // namespace display
