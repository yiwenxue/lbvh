#ifndef LBVH_HEADER_CUH
#define LBVH_HEADER_CUH

#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <math.h>
#include <memory>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__device__ inline int common_upper_bits(const unsigned int lhs, const unsigned int rhs) noexcept {
    return ::__clz(lhs ^ rhs);
}
__device__ inline int common_upper_bits(const unsigned long long int lhs, const unsigned long long int rhs) noexcept {
    return ::__clzll(lhs ^ rhs);
}

__device__ __host__ inline unsigned int expand_bits(unsigned int v) noexcept {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__ inline unsigned int morton_code(float4 xyz, float resolution = 1024.0f) noexcept {
    xyz.x                  = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f);
    xyz.y                  = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
    xyz.z                  = ::fminf(::fmaxf(xyz.z * resolution, 0.0f), resolution - 1.0f);
    const unsigned int xx = expand_bits(static_cast<unsigned int>(xyz.x));
    const unsigned int yy = expand_bits(static_cast<unsigned int>(xyz.y));
    const unsigned int zz = expand_bits(static_cast<unsigned int>(xyz.z));
    return xx * 4 + yy * 2 + zz;
}

__device__ __host__ inline unsigned int morton_code(double4 xyz, double resolution = 1024.0) noexcept {
    xyz.x                  = ::fmin(::fmax(xyz.x * resolution, 0.0), resolution - 1.0);
    xyz.y                  = ::fmin(::fmax(xyz.y * resolution, 0.0), resolution - 1.0);
    xyz.z                  = ::fmin(::fmax(xyz.z * resolution, 0.0), resolution - 1.0);
    const unsigned int xx = expand_bits(static_cast<unsigned int>(xyz.x));
    const unsigned int yy = expand_bits(static_cast<unsigned int>(xyz.y));
    const unsigned int zz = expand_bits(static_cast<unsigned int>(xyz.z));
    return xx * 4 + yy * 2 + zz;
}

struct aabb {
  float4 upper;
  float4 lower;
};

__device__ inline aabb aabb_merge (const aabb& lhs, const aabb& rhs) noexcept {
    return aabb{make_float4(::fminf(lhs.lower.x, rhs.lower.x), ::fminf(lhs.lower.y, rhs.lower.y), ::fminf(lhs.lower.z, rhs.lower.z), 0.0f),
                make_float4(::fmaxf(lhs.upper.x, rhs.upper.x), ::fmaxf(lhs.upper.y, rhs.upper.y), ::fmaxf(lhs.upper.z, rhs.upper.z), 0.0f)};
}

__device__ inline float4 aabb_center(const aabb& box) noexcept {
    float4 center;
    center.x = (box.lower.x + box.upper.x) * 0.5f;
    center.y = (box.lower.y + box.upper.y) * 0.5f;
    center.z = (box.lower.z + box.upper.z) * 0.5f;
    return center;
}

class Timer {
public:
  Timer(std::string name) : m_name(name) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
  }

  ~Timer() {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << m_name + " time: " << elapsedTime << "ms" << std::endl;
  }

private:
  std::string m_name;
  cudaEvent_t start, stop;
};

struct bvh_node {
  unsigned int parent_idx;
  unsigned int left_idx;
  unsigned int right_idx;
  unsigned int object_idx;
};

template <typename UInt>
__device__ inline uint2 bvh_node_range(UInt const        *node_code,
                                       const unsigned int num_leaves, unsigned int idx);

template <typename UInt>
__device__ inline unsigned int bvh_find_split(UInt const *node_code, const unsigned int num_leaves,
                                              const unsigned int first, const unsigned int last) noexcept;

template <typename UInt>
void bvh_construct_nodes(bvh_node          *nodes,
                         UInt const        *node_code,
                         const unsigned int num_objects);

template <typename Index, typename Buffer, typename AABBGetter,
          typename MortonCalculater>
class bvh {
  using buffer_t = Buffer;
  using index_t = Index;

public:
  template <typename InputIterator>
  bvh(InputIterator first, InputIterator last, const Buffer &buffer)
      : m_objs(first, last), m_buffer(buffer) {
    num_objs = m_objs.size();
    num_internal_nodes = num_objs - 1;
    num_nodes = num_objs * 2 - 1;
    construct();
  }

  void construct() {
    Timer total_timer("lbvh construction: total");

    const auto inf = std::numeric_limits<float>::infinity();
    aabb default_aabb;
    default_aabb.upper.x = -inf;
    default_aabb.lower.x = inf;
    default_aabb.upper.y = -inf;
    default_aabb.lower.y = inf;
    default_aabb.upper.z = -inf;
    default_aabb.lower.z = inf;

    {
        Timer timer("lbvh construction: compute aabb");
        m_aabbs.resize(num_nodes, default_aabb);
        const auto buffer_ptr = thrust::raw_pointer_cast(m_buffer.data());
        thrust::transform(
            thrust::device,
            m_objs.begin(),
            m_objs.end(),
            m_aabbs.begin(),
            AABBGetter(buffer_ptr)
        );
    }

    bool mortons_are_unique = false;
    thrust::device_vector<unsigned int> morton32;
    thrust::device_vector<unsigned int> indices;
    thrust::device_vector<unsigned long long> morton64;
    
    {
        Timer timer("lbvh construction: compute morton code");
        morton32.resize(num_objs);
        indices.resize(num_objs);
        morton64.resize(num_objs);
        const auto aabb_whole = thrust::reduce(
            thrust::device,
            m_aabbs.begin(),
            m_aabbs.end(),
            default_aabb,
            [] __device__ (const aabb& lhs, const aabb& rhs) {
                return aabb_merge(lhs, rhs);
            }
        );

        thrust::transform(
            thrust::device,
            m_aabbs.begin(),
            m_aabbs.end(),
            morton32.begin(),
            MortonCalculater(aabb_whole)
        );

        thrust::sequence(thrust::device, indices.begin(), indices.end());
        thrust::stable_sort_by_key(thrust::device,
            morton32.begin(), morton32.end(),
            thrust::make_zip_iterator(
                thrust::make_tuple(m_aabbs.begin() + num_internal_nodes, indices.begin()))
        );

        const auto unique = thrust::unique_copy(
            thrust::device,
            morton32.begin(), morton32.end(),
            morton64.begin()
        );
        mortons_are_unique = (unique == morton64.end());
        if (!mortons_are_unique) {
            thrust::transform(
                thrust::device,
                morton32.begin(), morton32.end(),
                indices.begin(),
                morton64.begin(),
                [] __device__ (const unsigned int m32, const unsigned int idx) {
                    unsigned long long int m64 = m32;
                    m64 <<= 32;
                    m64 |= idx;
                    return m64;
                }
            );
            morton32.clear();
        } else {
            morton64.clear();
        }
    }

    {
        Timer timer ("lbvh construction: radix tree");

        bvh_node default_node;
        default_node.parent_idx = 0xFFFFFFFF;
        default_node.left_idx   = 0xFFFFFFFF;
        default_node.right_idx  = 0xFFFFFFFF;
        default_node.object_idx = 0xFFFFFFFF;

        m_nodes.resize(num_nodes, default_node);
        thrust::transform(thrust::device,
            indices.begin(), indices.end(), 
            m_nodes.begin() + num_internal_nodes,
            [] __device__ (const unsigned int idx) {
                bvh_node node;
                node.parent_idx = 0xFFFFFFFF;
                node.left_idx   = 0xFFFFFFFF;
                node.right_idx  = 0xFFFFFFFF;
                node.object_idx = idx;
                return node;
            }
        );

        if (mortons_are_unique) {
            bvh_construct_nodes(
                thrust::raw_pointer_cast(m_nodes.data()),
                thrust::raw_pointer_cast(morton32.data()),
                num_objs
            );
        } else {
            bvh_construct_nodes(
                thrust::raw_pointer_cast(m_nodes.data()),
                thrust::raw_pointer_cast(morton64.data()),
                num_objs
            );
        }
    }

    {    
        Timer timer ("lbvh construction: internal aabbs");
        thrust::device_vector<int> flag_container(num_internal_nodes, 0);
        const auto flags = thrust::raw_pointer_cast(flag_container.data());
        const auto nodes = thrust::raw_pointer_cast(m_nodes.data());
        const auto aabbs = thrust::raw_pointer_cast(m_aabbs.data());

        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator(num_internal_nodes),
            thrust::make_counting_iterator(num_nodes),
            [aabbs, nodes, flags] __device__ (const unsigned int idx) {
                unsigned int parent_idx = nodes[idx].parent_idx;
                while (parent_idx != 0xFFFFFFFF) { // meant idx == 0
                    const int old = atomicCAS(flags + parent_idx, 0, 1);
                    if (old == 0) {
                        // this is the first thread entered here.
                        // wait the other thread from the other child node.
                        return;
                    }
                    assert(old == 1);

                    const auto lidx = nodes[parent_idx].left_idx;
                    const auto ridx = nodes[parent_idx].right_idx;
                    const auto rbox = aabbs[ridx];
                    const auto lbox = aabbs[lidx];
                    aabbs[parent_idx] = aabb_merge(lbox, rbox);
                    parent_idx = nodes[parent_idx].parent_idx;
                }
                return;
            }
        );
    }
  }

  void destruct() {}

private:
  unsigned int num_objs{0};
  unsigned int num_internal_nodes{0};
  unsigned int num_nodes{0};

  const Buffer &m_buffer;

  thrust::device_vector<index_t> m_objs;
  thrust::device_vector<aabb> m_aabbs;
  thrust::device_vector<bvh_node> m_nodes;
};

template <typename UInt>
__device__ inline uint2 bvh_node_range(UInt const        *node_code,
                                       const unsigned int num_leaves, unsigned int idx) {
    if (idx == 0) {
        return make_uint2(0, num_leaves - 1);
    }

    const UInt self_code = node_code[idx];
    const int  L_delta   = common_upper_bits(self_code, node_code[idx - 1]);
    const int  R_delta   = common_upper_bits(self_code, node_code[idx + 1]);
    const int  d         = (R_delta > L_delta) ? 1 : -1;

    const int delta_min = thrust::min(L_delta, R_delta);
    int       l_max     = 2;
    int       delta     = -1;
    int       i_tmp     = idx + d * l_max;
    if (0 <= i_tmp && i_tmp < num_leaves) {
        delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    while (delta > delta_min) {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves) {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
    }

    int l = 0;
    int t = l_max >> 1;
    while (t > 0) {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves) {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
        if (delta > delta_min) {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if (d < 0) {
        thrust::swap(idx, jdx);
    }
    return make_uint2(idx, jdx);
}

template <typename UInt>
__device__ inline unsigned int bvh_find_split(UInt const *node_code, const unsigned int num_leaves,
                                              const unsigned int first, const unsigned int last) noexcept {
    const UInt first_code = node_code[first];
    const UInt last_code  = node_code[last];
    if (first_code == last_code) {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits(first_code, last_code);

    int split  = first;
    int stride = last - first;
    do {
        stride           = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last) {
            const int delta = common_upper_bits(first_code, node_code[middle]);
            if (delta > delta_node) {
                split = middle;
            }
        }
    } while (stride > 1);

    return split;
}

template <typename UInt>
void bvh_construct_nodes(bvh_node          *nodes,
                         UInt const        *node_code,
                         const unsigned int num_objects) {
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(num_objects - 1),
        [nodes, node_code, num_objects] __device__(const unsigned int idx) {
            nodes[idx].object_idx = 0xFFFFFFFF;

            const uint2 ij    = bvh_node_range(node_code, num_objects, idx);
            const int   gamma = bvh_find_split(node_code, num_objects, ij.x, ij.y);

            nodes[idx].left_idx  = gamma;
            nodes[idx].right_idx = gamma + 1;
            if (thrust::min(ij.x, ij.y) == gamma) {
                nodes[idx].left_idx += num_objects - 1;
            }
            if (thrust::max(ij.x, ij.y) == gamma + 1) {
                nodes[idx].right_idx += num_objects - 1;
            }
            nodes[nodes[idx].left_idx].parent_idx  = idx;
            nodes[nodes[idx].right_idx].parent_idx = idx;
            return;
        });
    return;
}

#endif
