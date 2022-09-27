
#include <chrono>
#include <iostream>
#include <ratio>

#include "aabb.h"
#include "bvh.h"

#include "tiny_obj_loader.h"

#include "geometry.h"
#include "kernel.h"

using tbvh =
    bvh<int3, float4, triangle_mesh_aabb, default_morton_code_calculator>;

using Clock = std::chrono::high_resolution_clock;
using namespace std;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
    return 1;
  }

  auto timeStemp = Clock::now();

  // load the mesh from disk
  std::vector<std::shared_ptr<Vertices>> vertices_pool;
  std::vector<TriangleMesh> meshes;

  auto loader = [&](std::string filename) {
    vertices_pool.emplace_back();
    auto &vertices = vertices_pool.back();
    auto instances = loadMesh(filename, vertices);
    std::move(instances.begin(), instances.end(), std::back_inserter(meshes));
  };

  loader(std::string(argv[1]));
  // loader("/home/yiwenxue/program/spatial/dragon.obj");

  cout << "load mesh time: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() -
                                                                timeStemp)
              .count()
       << "ms" << endl;

  timeStemp = Clock::now();
  for (auto &mesh : meshes) {
    cout << "triangles: " << mesh.indices.size() << endl;
    const auto &pos = mesh.m_mesh->pos;
    tbvh bvh(mesh.indices.begin(), mesh.indices.end(), pos.begin(), pos.end());
  }

  thrust::host_vector<int3> h_indices(100);
  thrust::host_vector<float4> vertex(100);

  cout << "bvh construct time: "
       << std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() -
                                                                timeStemp)
              .count()
       << "us" << endl;

  return 0;
}
