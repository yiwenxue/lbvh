#include "frame.h"
#include "window.h"

#include "loader/loader.h"

#include <iostream>
#include <vector>

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

Shader  *Frame::m_shader = nullptr;
GLuint   Frame::m_vao    = 0;
uint32_t Frame::m_count  = 0;

void Frame::initialize() {
    if (m_count == 0) {
        m_shader = new Shader(shaderStages);
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);
    }

    glGenTextures(1, &m_glTexture);
    glBindTexture(GL_TEXTURE_2D, m_glTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_dim.width, m_dim.height, 0,
                 GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}

Frame::Frame(const Extent2D &size) : m_dim{size} {
    initialize();
    m_count++;
}

Frame::Frame(uint32_t width, uint32_t height) : m_dim{width, height} {
    initialize();
    m_count++;
}

Frame::~Frame() {
    glDeleteTextures(1, &m_glTexture);
    m_count--;
    if (m_count == 0) {
        delete m_shader;
        glDeleteVertexArrays(1, &m_vao);
    }
}

void Frame::present() const {
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
        rgba[i * 4]     = rgb[i * 3];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    return rgba;
}
} // namespace

void Frame::load(const std::string &path) {
    int width, height, channels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char *data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    stbi_flip_vertically_on_write(false);
    if (data) {
        m_dim.width  = width;
        m_dim.height = height;
        glBindTexture(GL_TEXTURE_2D, m_glTexture);
        if (channels == 3) {
            unsigned char *rgba = rgb2rgba(data, width, height);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, rgba);
            delete[] rgba;
        } else {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, data);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        stbi_image_free(data);
    } else {
        std::cerr << "Failed to load texture: " + path << std::endl;
    }
}

void Frame::save(const std::string &path) const {
    std::vector<unsigned char> data(m_dim.width * m_dim.height * 4);
    glBindTexture(GL_TEXTURE_2D, m_glTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    stbi_flip_vertically_on_write(true);
    stbi_write_png(path.c_str(), m_dim.width, m_dim.height, 4, data.data(),
                   m_dim.width * 4);
    stbi_flip_vertically_on_write(false);
}

void Frame::resize(uint32_t width, uint32_t height) {
    m_dim.width  = width;
    m_dim.height = height;
    glBindTexture(GL_TEXTURE_2D, m_glTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA,
                 GL_FLOAT, nullptr);
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
