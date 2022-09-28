#pragma once
#include "glad/glad.h"
#include "shader.h"
#include "window.h"
#include <algorithm>

namespace display {

/**
 * @brief A basic frame which have the ability to interop with CUDA
 * @details A frame can be used to display a texture on the screen. It can also
 * be write to by CUDA. There are two ways to write to a frame. The first is to
 * upload contents from a device buffer to the frame. The second is to write to
 * a cudaSurfaceObject_t using surf2Dwrite.
 */
class Frame {
public:
  Frame(int width, int height);
  Frame(const int2 &size);

  ~Frame();

  void present();

  cudaSurfaceObject_t getSurf() noexcept { return m_cudaSurf; }

  void load(const float4 *data, int width, int height);

  void save(const std::string &path) const;

  void resize(int width, int height);

  void resize(const int2 &size) { resize(size.x, size.y); }

  int2 getSize() const noexcept { return m_dim; }

  void clear();

private:
  void initialize();

  int2 m_dim;
  GLuint m_glTexture;
  static Shader *m_shader;
  static GLuint m_vao;

  cudaGraphicsResource_t m_cu_texture;
  cudaSurfaceObject_t m_cudaSurf;
  cudaArray_t m_cudaArray;

  static int m_count;
};

}; // namespace display
