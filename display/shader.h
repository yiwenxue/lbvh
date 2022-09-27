#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>
namespace display {
enum class ShaderStageType {
    None                  = 0,
    Vertex                = 1,
    Fragment              = 2,
    Geometry              = 3,
    TessalationControl    = 4,
    TessalationEvaluation = 5,
    Compute               = 6,
};

struct ShaderStage {
    ShaderStageType type;
    std::string     source;
};

class Shader {
public:
    unsigned int ID;

    Shader(std::vector<ShaderStage> stages);

    void use();
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;

    void setVec2(const std::string &name, const glm::vec2 &value) const;
    void setVec2(const std::string &name, float x, float y) const;
    void setVec3(const std::string &name, const glm::vec3 &value) const;
    void setVec3(const std::string &name, float x, float y, float z) const;
    void setVec4(const std::string &name, const glm::vec4 &value) const;
    void setVec4(const std::string &name, float x, float y, float z, float w);
    void setMat2(const std::string &name, const glm::mat2 &mat) const;
    void setMat3(const std::string &name, const glm::mat3 &mat) const;
    void setMat4(const std::string &name, const glm::mat4 &mat) const;

    void setUniformBlockBinding(const std::string &name,
                                unsigned int       binding) const;

private:
    void checkCompileErrors(unsigned int shader, std::string type);
};
} // namespace display
