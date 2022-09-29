#include "bvh.h"
#include "utils.h"

#include <cassert>
#include <cstring>
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

template <typename Index, typename Buffer, typename AABBGetter,
          typename MortonCalculater>
void bvh<Index, Buffer, AABBGetter, MortonCalculater>::construct() {
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
            morton32.shrink_to_fit();
        } else {
            morton64.clear();
            morton64.shrink_to_fit();
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

    // check if all nodes are constructed correctly.
    {
        const auto limit = num_nodes;

        // check if all internal nodes have two children and one parent.
        thrust::device_vector<int> flag_container(num_nodes, 0);
        const auto flags = thrust::raw_pointer_cast(flag_container.data());
        const auto nodes = thrust::raw_pointer_cast(m_nodes.data());

        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator<unsigned int>(num_internal_nodes),
            thrust::make_counting_iterator<unsigned int>(num_nodes),
            [flags, nodes, limit] __device__ (const unsigned int idx) {
                const auto node = nodes[idx];
                const auto lidx = node.left_idx;
                const auto ridx = node.right_idx;
                const auto pidx = node.parent_idx;
                const auto oidx = node.object_idx;
                if ((lidx != 0xFFFFFFFF) || (ridx != 0xFFFFFFFF) || (oidx == 0xFFFFFFFF) || (oidx >= limit)
                    || (pidx == 0xFFFFFFFF) || (pidx >= limit)) {
                    atomicAdd(flags + idx, 1);
                }
            }
        );

        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator<unsigned int>(1),
            thrust::make_counting_iterator<unsigned int>(num_internal_nodes),
            [flags, nodes, limit] __device__ (const unsigned int idx) {
                const auto node = nodes[idx];
                const auto lidx = node.left_idx;
                const auto ridx = node.right_idx;
                const auto pidx = node.parent_idx;
                const auto oidx = node.object_idx;
                if ((lidx == 0xFFFFFFFF) || (ridx == 0xFFFFFFFF) || (pidx == 0xFFFFFFFF) || (oidx != 0xFFFFFFFF)
                    || (lidx >= limit) || (ridx >= limit) || (pidx >= limit)) {
                    atomicAdd(flags + idx, 1);
                }
            }
        );

        const auto res = thrust::reduce(
            thrust::device,
            flag_container.begin(), flag_container.end()
        );

        if (res != 0) {
            std::cerr << "Error: invalid BVH construction. errors: " << res << std::endl;
            std::exit(1);
        }
    }

    {
        // show the aabbs
        thrust::host_vector<aabb> aabbs(m_aabbs);
        for (unsigned int i = 0; i < num_nodes; ++i) {
            const auto& box = aabbs[i];
            std::cout << "node " << i << ": " << box.lower.x << " - " << box.upper.x << std::endl;
        }
    }
}

#include "kernel.h"
