#include "frame.h"
#include "window.h"

#include "loader/loader.h"

#include <iostream>
#include <vector>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

namespace display {

namespace {
const std::string fullscreen_quad_vs = R"(
#version 330 core

const vec4 pos[4] = vec4[4](
	vec4(-1, 1, 0.5, 1),
	vec4(-1, -1, 0.5, 1),
	vec4(1, 1, 0.5, 1),
	vec4(1, -1, 0.5, 1)
);

const vec2 uv[4] = vec2[4](
	vec2(0, 1),
	vec2(0, 0),
	vec2(1, 1),
	vec2(1, 0)
);

out vec2 texcoord;

void main(void){
	gl_Position = pos[gl_VertexID];
	texcoord = uv[gl_VertexID];
}
)";

const std::string display_texture_fs = R"(
#version 330 core

uniform sampler2D img;

in vec2 texcoord;

out vec4 color;

void main(void){
	color = texture(img, texcoord);
})";

const std::vector<ShaderStage> shaderStages{
    ShaderStage{ShaderStageType::Vertex, fullscreen_quad_vs},
    ShaderStage{ShaderStageType::Fragment, display_texture_fs},
};
} // namespace

Shader *Frame::m_shader = nullptr;
GLuint Frame::m_vao = 0;
int Frame::m_count = 0;

void Frame::initialize() {
  if (m_count == 0) {
    m_shader = new Shader(shaderStages);
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);
  }

  glGenTextures(1, &m_glTexture);
  glBindTexture(GL_TEXTURE_2D, m_glTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_dim.x, m_dim.y, 0, GL_RGBA,
               GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  cudaGraphicsGLRegisterImage(&m_cu_texture, m_glTexture, GL_TEXTURE_2D,
                              cudaGraphicsRegisterFlagsNone);
  cudaGraphicsMapResources(1, &m_cu_texture, 0);
  cudaGraphicsSubResourceGetMappedArray(&m_cudaArray, m_cu_texture, 0, 0);
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = m_cudaArray;
  cudaCreateSurfaceObject(&m_cudaSurf, &resDesc);
  cudaGraphicsUnmapResources(1, &m_cu_texture, 0);

  glBindTexture(GL_TEXTURE_2D, 0);
}

Frame::Frame(const int2 &size) : m_dim{size} {
  initialize();
  m_count++;
}

Frame::Frame(int width, int height) : m_dim{make_int2(width, height)} {
  initialize();
  m_count++;
}

Frame::~Frame() {
  glDeleteTextures(1, &m_glTexture);
  assert(m_count > 0);
  m_count--;
  if (m_count == 0) {
    delete m_shader;
    glDeleteVertexArrays(1, &m_vao);
  }
}

void Frame::present() {
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);

  m_shader->use();

  glBindVertexArray(m_vao);
  glBindTexture(GL_TEXTURE_2D, m_glTexture);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindVertexArray(0);
}

namespace {
// a util function to convert rgb to rgba
unsigned char *rgb2rgba(unsigned char *rgb, int width, int height) {
  unsigned char *rgba = new unsigned char[width * height * 4];
  for (int i = 0; i < width * height; i++) {
    rgba[i * 4] = rgb[i * 3];
    rgba[i * 4 + 1] = rgb[i * 3 + 1];
    rgba[i * 4 + 2] = rgb[i * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}
} // namespace

void Frame::load(const float4 *data, int width, int height) {
  assert(width == m_dim.x && height == m_dim.y);
  cudaGraphicsMapResources(1, &m_cu_texture, 0);
  cudaArray_t cuArray;
  cudaGraphicsSubResourceGetMappedArray(&cuArray, m_cu_texture, 0, 0);
  cudaMemcpy2DToArray(cuArray, 0, 0, data, width * sizeof(float4),
                      width * sizeof(float4), height, cudaMemcpyDeviceToDevice);
  cudaGraphicsUnmapResources(1, &m_cu_texture, 0);
}

void Frame::save(const std::string &path) const {
  std::vector<unsigned char> data(m_dim.x * m_dim.y * 4);
  glBindTexture(GL_TEXTURE_2D, m_glTexture);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
  glBindTexture(GL_TEXTURE_2D, 0);
  stbi_flip_vertically_on_write(true);
  stbi_write_png(path.c_str(), m_dim.x, m_dim.y, 4, data.data(), m_dim.x * 4);
  stbi_flip_vertically_on_write(false);
}

void Frame::resize(int width, int height) {
  m_dim.x = width;
  m_dim.y = height;
  glBindTexture(GL_TEXTURE_2D, m_glTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA,
               GL_FLOAT, nullptr);

  cudaGraphicsGLRegisterImage(&m_cu_texture, m_glTexture, GL_TEXTURE_2D,
                              cudaGraphicsRegisterFlagsNone);
  cudaGraphicsMapResources(1, &m_cu_texture, 0);
  cudaGraphicsSubResourceGetMappedArray(&m_cudaArray, m_cu_texture, 0, 0);
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = m_cudaArray;
  cudaCreateSurfaceObject(&m_cudaSurf, &resDesc);
  cudaGraphicsUnmapResources(1, &m_cu_texture, 0);

  glBindTexture(GL_TEXTURE_2D, 0);
}

void Frame::clear() {
  GLuint fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         m_glTexture, 0);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
}

} // namespace display
