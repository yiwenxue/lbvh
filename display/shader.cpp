#include "shader.h"

#include "glad/glad.h"

#include <algorithm>
#include <cassert>
#include <vector>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace display {

const std::string shaderTypeString[] = {
    "None",
    "Vertex",
    "Fragment",
    "Geometry",
    "TessalationControl",
    "TessalationEvaluation",
    "Compute",
};

const GLenum ShaderStageTypeGL[] = {
    GL_NONE,
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    GL_GEOMETRY_SHADER,
    GL_TESS_CONTROL_SHADER,
    GL_TESS_EVALUATION_SHADER,
    GL_COMPUTE_SHADER,
};

Shader::Shader(std::vector<ShaderStage> stages) {
  if (stages.empty()) {
    assert(false && "Shader stages are empty");
  }
  std::sort(stages.begin(), stages.end(),
            [](const ShaderStage &a, const ShaderStage &b) {
              return a.type < b.type;
            });

  ID = glCreateProgram();

  std::vector<unsigned int> shaderIDs;

  for (auto &stage : stages) {
    unsigned int shaderID =
        glCreateShader(ShaderStageTypeGL[static_cast<int>(stage.type)]);
    shaderIDs.push_back(shaderID);
    const char *source = stage.source.c_str();
    glShaderSource(shaderID, 1, &source, NULL);
    glCompileShader(shaderID);
    checkCompileErrors(shaderID,
                       shaderTypeString[static_cast<int>(stage.type)]);
    glAttachShader(ID, shaderID);
  }

  glLinkProgram(ID);
  checkCompileErrors(ID, "PROGRAM");

  // delete the shaders as they're linked into our program now and no longer
  // necessery
  for (auto &shaderID : shaderIDs) {
    glDeleteShader(shaderID);
  }
}

void Shader::use() { glUseProgram(ID); }
void Shader::setBool(const std::string &name, bool value) const {
  glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
}
void Shader::setInt(const std::string &name, int value) const {
  glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}
void Shader::setFloat(const std::string &name, float value) const {
  glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::setVec2(const std::string &name, const glm::vec2 &value) const {
  glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}

void Shader::setVec2(const std::string &name, float x, float y) const {
  glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
}

void Shader::setVec3(const std::string &name, const glm::vec3 &value) const {
  glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}

void Shader::setVec3(const std::string &name, float x, float y, float z) const {
  glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
}

void Shader::setVec4(const std::string &name, const glm::vec4 &value) const {
  glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}

void Shader::setVec4(const std::string &name, float x, float y, float z,
                     float w) {
  glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
}

void Shader::setMat2(const std::string &name, const glm::mat2 &mat) const {
  glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE,
                     &mat[0][0]);
}

void Shader::setMat3(const std::string &name, const glm::mat3 &mat) const {
  glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE,
                     &mat[0][0]);
}

void Shader::setMat4(const std::string &name, const glm::mat4 &mat) const {
  glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE,
                     &mat[0][0]);
}

void Shader::setUniformBlockBinding(const std::string &name,
                                    unsigned int binding) const {
  uint32_t blockIndex = glGetUniformBlockIndex(ID, name.c_str());
  glUniformBlockBinding(ID, blockIndex, binding);
}

void Shader::checkCompileErrors(unsigned int shader, std::string type) {
  int success;
  char infoLog[2048] = {};
  if (type != "PROGRAM") {
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(shader, 2048, NULL, infoLog);
      std::cout
          << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n"
          << infoLog
          << "\n -- --------------------------------------------------- -- "
          << std::endl;
    }
  } else {
    glGetProgramiv(shader, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(shader, 2048, NULL, infoLog);
      std::cout
          << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n"
          << infoLog
          << "\n -- --------------------------------------------------- -- "
          << std::endl;
    }
  }
}

} // namespace display
