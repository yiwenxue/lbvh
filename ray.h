#pragma once

#include "helper_math.h"

/**
 * @brief A simple ray class
 *
 */
struct ray {
  float4 origin;
  float4 direction;
};

#include "../bvh.h"
#include "../camera.h"

// ray nearest hit
template <typename Object, typename BufferType>
bvh_node *rayNearestHit(const ray &r,
                        const bvh_tree<Object, BufferType, true> &root) {
  bvh_node *nodes = root.nodes;
}
