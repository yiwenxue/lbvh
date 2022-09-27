#pragma once
#include "glad/glad.h"
#include "shader.h"
#include "types.h"
#include "window.h"
#include <algorithm>

namespace display {

// Frame is infact a gltexture, its texture unit will be passed to cuda to
// register at cuda. when cuda finished its calculation, the program will call a
// present, and then the content will be drawn using a simple shader
class Frame {
public:
  Frame(uint32_t width, uint32_t height);
  Frame(const Extent2D &size);

  ~Frame();

  GLuint getHandle() const noexcept { return m_glTexture; }

  void present() const;

  void load(const std::string &path);

  void save(const std::string &path) const;

  void resize(uint32_t width, uint32_t height);

  void resize(const Extent2D &size) { resize(size.width, size.height); }

  Extent2D getSize() const noexcept { return m_dim; }

  void clear();

private:
  void initialize();

  Extent2D m_dim;
  GLuint m_glTexture;
  static Shader *m_shader;
  static GLuint m_vao;
  static uint32_t m_count;
};

}; // namespace display
