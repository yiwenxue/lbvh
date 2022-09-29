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
