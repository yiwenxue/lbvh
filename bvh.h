#ifndef BVH_HEADER_H
#define BVH_HEADER_H

#include "aabb.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct bvh_node {
  unsigned int parent_idx;
  unsigned int left_idx;
  unsigned int right_idx;
  unsigned int object_idx;
};

template <typename Object, typename BufferType, bool isConst> struct bvh_tree;

template <typename Object, typename BufferType>
struct bvh_tree<Object, BufferType, false> {
  uint32_t num_objs;
  uint32_t num_nodes;

  bvh_node *nodes;
  aabb *aabbs;
  Object *objects;
  BufferType *buffer;
};

template <typename Object, typename BufferType>
struct bvh_tree<Object, BufferType, true> {
  const uint32_t num_objs;
  const uint32_t num_nodes;

  const bvh_node *nodes;
  const aabb *aabbs;
  const Object *objects;
  const BufferType *buffer;
};

template <typename Index, typename BufferType, typename AABBGetter,
          typename MortonCalculater>
class bvh {
  using buffer_t = BufferType;
  using index_t = Index;

public:
  template <typename InputIterator, typename BufferIterator>
  bvh(InputIterator first, InputIterator last, BufferIterator bfirst,
      BufferIterator blast)
      : m_objs(first, last), m_buffer(bfirst, blast) {
    num_objs = m_objs.size();
    num_internal_nodes = num_objs - 1;
    num_nodes = num_objs * 2 - 1;
    construct();
  }

  bvh_tree<index_t, buffer_t, false> get_tree() {
    return bvh_tree<index_t, buffer_t, false>{num_objs,
            num_nodes,
            thrust::raw_pointer_cast(m_nodes.data()),
            thrust::raw_pointer_cast(m_aabbs.data()),
            thrust::raw_pointer_cast(m_objs.data()),
            thrust::raw_pointer_cast(m_buffer.data())};
  }

  bvh_tree<index_t, buffer_t, true> get_ctree() const {
    return bvh_tree<index_t, buffer_t, true>{num_objs,
            num_nodes,
            thrust::raw_pointer_cast(m_nodes.data()),
            thrust::raw_pointer_cast(m_aabbs.data()),
            thrust::raw_pointer_cast(m_objs.data()),
            thrust::raw_pointer_cast(m_buffer.data())};
  }

  void construct();

  void destruct() {}

private:
  unsigned int num_objs{0};
  unsigned int num_internal_nodes{0};
  unsigned int num_nodes{0};

  thrust::device_vector<buffer_t> m_buffer;
  thrust::device_vector<index_t> m_objs;
  thrust::device_vector<aabb> m_aabbs;
  thrust::device_vector<bvh_node> m_nodes;
};

template <typename Index, typename BufferType>
bool bvh_tree_validator(const bvh_tree<Index, BufferType, true> &tree) {
  return true;
}

#include <iostream>
#include <fstream>

std::ostream &operator<<(std::ostream &os, const float3 &v);
std::ostream &operator<<(std::ostream &os, const float4 &v);
std::ostream &operator<<(std::ostream &os, const aabb &v);

template <typename Index, typename BufferType>
void bvh_tree_visualizer(const bvh_tree<Index, BufferType, true> &tree) {
  auto num_nodes = tree.num_nodes;
    thrust::host_vector<aabb> aabbs(tree.aabbs, tree.aabbs + num_nodes);
    thrust::host_vector<bvh_node> nodes(tree.nodes, tree.nodes + num_nodes);

    std::cout << "size: " << aabbs.size() << std::endl;

    // visualize the tree structure using graphviz
    std::ofstream ofs("lbvh.dot");
    ofs << "digraph G {" << std::endl;
    for (unsigned int i = 0; i < num_nodes; ++i) {
        const auto& node = nodes[i];
        const auto& aabb = aabbs[i];
        // for leaf nodes, render as a box
        if (node.object_idx != 0xFFFFFFFF) {
            ofs << "  " << i << " [label=\"" << i << "\\nupper" << aabb.upper << "\\nlower" << aabb.lower << "\\n" << node.object_idx << "\"];" << std::endl;
        } else {
            ofs << "  " << i << " [label=\"" << i << "\\nupper" << aabb.upper << "\\nlower" << aabb.lower << "\"];" << std::endl;
            // links
            ofs << i << " -> " << node.left_idx << ";" << std::endl;
            ofs << i << " -> " << node.right_idx << ";" << std::endl;
        }
    }
    ofs << "}" << std::endl;
}

#endif
