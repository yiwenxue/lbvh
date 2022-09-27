#ifndef GEOMETRY_HEADER_H
#define GEOMETRY_HEADER_H

#include "helper_math.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Vertices {
  thrust::host_vector<float4> pos;
  thrust::host_vector<float4> norm;
  thrust::host_vector<float2> uv;

  Vertices() = default;

  Vertices(Vertices &&other) noexcept
      : pos(std::move(other.pos)), norm(std::move(other.norm)),
        uv(std::move(other.uv)) {}

  Vertices &operator=(Vertices &&other) noexcept {
    pos = std::move(other.pos);
    norm = std::move(other.norm);
    uv = std::move(other.uv);
    return *this;
  }

  Vertices(const Vertices &other) = delete;
  Vertices &operator=(const Vertices &other) = delete;

  ~Vertices() = default;
};

struct TriangleMesh {
  thrust::host_vector<int3> indices;
  std::shared_ptr<Vertices> m_mesh;

  TriangleMesh() = default;

  TriangleMesh(TriangleMesh &&other) noexcept
      : indices(std::move(other.indices)), m_mesh(other.m_mesh) {}

  TriangleMesh &operator=(TriangleMesh &&other) noexcept {
    indices = std::move(other.indices);
    m_mesh = other.m_mesh;
    return *this;
  }

  TriangleMesh(const TriangleMesh &other) = delete;
  TriangleMesh &operator=(const TriangleMesh &other) = delete;

  ~TriangleMesh() = default;
};

#include "loader/loader.h"

std::vector<TriangleMesh> loadMesh(const std::string &filename,
                                   std::shared_ptr<Vertices> &vertices) {
  std::vector<TriangleMesh> meshes;
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn;
  std::string err;

  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                              filename.c_str());

  if (vertices.get() == nullptr) {
    vertices = std::make_shared<Vertices>();
  }

  if (!warn.empty()) {
    std::cout << warn << std::endl;
  }

  if (!err.empty()) {
    std::cerr << err << std::endl;
  }

  if (!ret) {
    std::cerr << "Failed to load " << filename << std::endl;
    return meshes;
  }

  thrust::host_vector<float4> h_pos(attrib.vertices.size() / 3);

  { // copy the positions
    for (size_t i = 0; i < attrib.vertices.size() / 3; i++) {
      h_pos[i] =
          make_float4(attrib.vertices[3 * i + 0], attrib.vertices[3 * i + 1],
                      attrib.vertices[3 * i + 2], 1.0f);
    }
    vertices->pos = h_pos;
  }

  { // copy the normals
    thrust::host_vector<float4> h_norm(attrib.normals.size() / 3);
    for (size_t i = 0; i < attrib.normals.size() / 3; i++) {
      h_norm[i] =
          make_float4(attrib.normals[3 * i + 0], attrib.normals[3 * i + 1],
                      attrib.normals[3 * i + 2], 0.0f);
    }
    vertices->norm = h_norm;
  }

  { // copy the uvs
    thrust::host_vector<float2> h_uv(attrib.texcoords.size() / 2);
    for (size_t i = 0; i < attrib.texcoords.size() / 2; i++) {
      h_uv[i] =
          make_float2(attrib.texcoords[2 * i + 0], attrib.texcoords[2 * i + 1]);
    }
    vertices->uv = h_uv;
  }

  // copy the indices
  for (size_t i = 0; i < shapes.size(); i++) {
    TriangleMesh mesh;
    thrust::host_vector<int3> h_indices(shapes[i].mesh.indices.size() / 3);

    for (size_t j = 0; j < shapes[i].mesh.indices.size() / 3; j++) {
      h_indices[j] = make_int3(shapes[i].mesh.indices[3 * j + 0].vertex_index,
                               shapes[i].mesh.indices[3 * j + 1].vertex_index,
                               shapes[i].mesh.indices[3 * j + 2].vertex_index);
    }

    mesh.indices = h_indices;
    mesh.m_mesh = vertices;
    meshes.push_back(std::move(mesh));
  }
  return meshes;
}

#endif
