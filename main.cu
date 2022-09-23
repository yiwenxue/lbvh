#include "lbvh.cuh"
#include "tiny_obj_loader.h"

#include <chrono>
#include <iostream>
#include <ratio>


struct Vertices {
  thrust::device_vector<float4> pos;
  thrust::device_vector<float4> norm;
  thrust::device_vector<float2> uv;

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

struct Triangles {
  thrust::device_vector<int3> indices;
  std::shared_ptr<Vertices> m_mesh;

  Triangles() = default;

  Triangles(Triangles &&other) noexcept
      : indices(std::move(other.indices)), m_mesh(other.m_mesh) {}

  Triangles &operator=(Triangles &&other) noexcept {
    indices = std::move(other.indices);
    m_mesh = other.m_mesh;
    return *this;
  }

  Triangles(const Triangles &other) = delete;
  Triangles &operator=(const Triangles &other) = delete;

  ~Triangles() = default;
};

// load the obj mesh from file
std::vector<Triangles> loadMesh(const std::string &filename,
                                std::shared_ptr<Vertices> &vertices) {
  std::vector<Triangles> meshes;
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
    Triangles mesh;
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

using Clock = std::chrono::high_resolution_clock;
using namespace std;

struct triangle_mesh_aabb {
  triangle_mesh_aabb() = default;
  triangle_mesh_aabb(const float4 *buffer) : m_buffer(buffer) {}
  __device__ aabb operator()(int3 idx) const noexcept {
    aabb box;
    const float4 &v0 = m_buffer[idx.x];
    const float4 &v1 = m_buffer[idx.y];
    const float4 &v2 = m_buffer[idx.z];

    box.lower.x = fminf(fminf(v0.x, v1.x), v2.x);
    box.lower.y = fminf(fminf(v0.y, v1.y), v2.y);
    box.lower.z = fminf(fminf(v0.z, v1.z), v2.z);

    box.upper.x = fmaxf(fmaxf(v0.x, v1.x), v2.x);
    box.upper.y = fmaxf(fmaxf(v0.y, v1.y), v2.y);
    box.upper.z = fmaxf(fmaxf(v0.z, v1.z), v2.z);

    return box;
  };
  const float4 *m_buffer;
};

struct default_morton_code_calculator {
  default_morton_code_calculator(aabb w) : whole(w) {}
  default_morton_code_calculator() = default;
  ~default_morton_code_calculator() = default;
  default_morton_code_calculator(default_morton_code_calculator const &) =
      default;
  default_morton_code_calculator(default_morton_code_calculator &&) = default;
  default_morton_code_calculator &
  operator=(default_morton_code_calculator const &) = default;
  default_morton_code_calculator &
  operator=(default_morton_code_calculator &&) = default;

  __device__ unsigned int operator()(const aabb &box) noexcept {
    auto p = aabb_center(box);
    p.x -= whole.lower.x;
    p.y -= whole.lower.y;
    p.z -= whole.lower.z;
    p.x /= (whole.upper.x - whole.lower.x);
    p.y /= (whole.upper.y - whole.lower.y);
    p.z /= (whole.upper.z - whole.lower.z);
    return morton_code(p);
  }
  aabb whole;
};

using lbvh =  bvh<int3, thrust::device_vector<float4>, triangle_mesh_aabb,
          default_morton_code_calculator>;

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
    return 1;
  }

  auto timeStemp = Clock::now();

  // load the mesh from disk
  std::vector<std::shared_ptr<Vertices>> vertices_pool;
  std::vector<Triangles> meshes;

  auto loader = [&](std::string filename) {
    vertices_pool.emplace_back();
    auto &vertices = vertices_pool.back();
    auto instances = loadMesh(filename, vertices);
    std::move(instances.begin(), instances.end(), std::back_inserter(meshes));
  };

  loader(std::string(argv[1]));

  cout << "load mesh time: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                                timeStemp)
              .count()
       << "ms" << endl;

  timeStemp = Clock::now();
  for (auto &mesh : meshes) {
    cout << "triangles: " << mesh.indices.size() << endl;
    // build the lbvh
    lbvh bvh(mesh.indices.begin(), mesh.indices.end(), mesh.m_mesh->pos);
  }

  cout << "bvh construct time: "
       << std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                                timeStemp)
              .count()
       << "us" << endl;

  return 0;
}
