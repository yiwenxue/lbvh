#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>
#include <iostream>

#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923

__global__ void kernel(float4 *framebuffer, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    int index = y * width + x;
    framebuffer[index] = make_float4(x / (float)width, y / (float)height, 0.2f, 1.0f);
}

__global__ void kernelSurf(cudaSurfaceObject_t surf, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    surf2Dwrite(make_float4(x / (float)width, y / (float)height, 0.2f, 1.0f), surf, x * sizeof(float4), y, cudaBoundaryModeZero);
}

void renderer(float4 *framebuffer, int width, int height)
{
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    kernel<<<grid, block>>>(framebuffer, width, height);
}

void renderSurf(cudaSurfaceObject_t surf, int width, int height)
{
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    kernelSurf<<<grid, block>>>(surf, width, height);
}

#include "../camera.h"
#include "../ray.h"

/**
 * @brief Get the Ray object
 *
 * @param cam the camera, which is used to generate the ray
 * @param uv the uv coordinates, where (0,0) is the bottom left corner and (1,1)
 * @param jitter the jitter, which is used to create a depth of field effect
 * @return ray
 */
__device__ __host__ ray getCamRay(const camera *cam, float2 uv, float2 jitter)
{
    float3 pos = make_float3(cam->pos);
    float3 view = normalize(make_float3(cam->view));
    float3 up = make_float3(cam->up);

    float3 right = normalize(cross(view, up));
    float3 up2 = normalize(cross(right, view));

    float3 dir = normalize(
        view * cam->focalDist +
        right * (uv.x - 0.5f) * tan(cam->fov.x / 360.0 * M_PI) +
        up2 * (uv.y - 0.5f) * tan(cam->fov.y / 360.0 * M_PI));

    float3 jitteredPos =
        pos + right * jitter.x * cam->aperture + up2 * jitter.y * cam->aperture;

    return ray{
        make_float4(jitteredPos, 0.0f),
        make_float4(normalize(jitteredPos + dir * cam->focalDist - pos), 0.0f)};
}

__device__ float intersectSphere(ray mray, const float4 sphere)
{
    float3 o = make_float3(mray.origin);
    float3 d = make_float3(mray.direction);
    float3 center = make_float3(sphere);
    float radius = sphere.w;

    float3 oc = o - center;
    float a = dot(d, d);
    float half_b = dot(oc, d);
    float c = dot(oc, oc) - radius * radius;

    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0)
    {
        return -1.0f;
    }
    else
    {
        return (-half_b - sqrt(discriminant)) / a;
    }
}

__global__ void traceRay(cudaSurfaceObject_t surf, int width, int height, const camera *cam, const float4 *spheres, int sphereCount)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > width || y > height)
        return;

    // generate ray
    float u = x / (float)width;
    float v = y / (float)height;
    float2 uv = make_float2(u, v);
    float2 jitter = make_float2(0.0f, 0.0f);
    ray mray = getCamRay(cam, uv, jitter);

    float depth = 1e20f;
    for (int i = 0; i < sphereCount; i++)
    {
        float t = intersectSphere(mray, spheres[i]);
        if (t > 0.0f && t < depth)
            depth = t;
    }

    float far = 50.0f;
    float near = 0.1f;

    depth = (depth - near) / (far - near);

    surf2Dwrite(make_float4(depth, depth, depth, 1.0f), surf, x * sizeof(float4), y, cudaBoundaryModeZero);
}

constexpr int SPHERE_COUNT = 20;

struct SphereGen
{
    thrust::default_random_engine rng;
    thrust::normal_distribution<float> dist;

    SphereGen(float mean, float stddev) : dist(mean, stddev) {}

    __host__ __device__ float4 operator()(const int &i)
    {
        rng.discard(i);
        float3 pos = make_float3(dist(rng), 0, dist(rng));
        float radius = 1.0f + 0.5f * dist(rng);
        return make_float4(pos, radius);
    }
};

void render_frame(cudaSurfaceObject_t surf, int width, int height)
{
    // generate spheres
    thrust::device_vector<float4> spheres(SPHERE_COUNT);

    thrust::transform(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(SPHERE_COUNT), spheres.begin(), SphereGen(0.0f, 5.0f));
    float4 base_sphere = make_float4(0.0f, -1000.0f - 2.0f, 0.0f, 1000.0f);
    spheres[0] = base_sphere;

    camera *cam_d = nullptr;
    cudaMalloc(&cam_d, sizeof(camera));
    // generate camera
    float fov = 45.0f;
    camera cam{
        make_float4(0, 5, -10, 0),
        make_float4(0, -0.2, 1, 0),
        make_float4(0, 1, 0, 0),
        make_float2(fov, fov * height / width),
        0.01f,
        0.1f,
    };
    cudaMemcpy(cam_d, &cam, sizeof(camera), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    const camera *pCam = cam_d;
    const float4 *pSpheres = thrust::raw_pointer_cast(spheres.data());

    // loop for each pixel
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    traceRay<<<grid, block>>>(surf, width, height, pCam, pSpheres, SPHERE_COUNT);

    cudaFree(cam_d);
}

#include "../intersection.h"
#include "bvh.h"

__device__ float intersectTriangle(ray iray, int3 index, const float4 *buffer) {
    float4 v0 = buffer[index.x];
    float4 v1 = buffer[index.y];
    float4 v2 = buffer[index.z];

    // ray-triangle intersection
    float3 e1 = make_float3(v1) - make_float3(v0);
    float3 e2 = make_float3(v2) - make_float3(v0);
    float3 s1 = cross(make_float3(iray.direction), e2);

    float divisor = dot(s1, e1);
    if (divisor == 0.0f)
        return -1.0f;

    float3 d = make_float3(iray.origin) - make_float3(v0);
    float b1 = dot(d, s1) / divisor;
    if (b1 < 0.0f || b1 > 1.0f)
        return -1.0f;

    float3 s2 = cross(d, e1);
    float b2 = dot(make_float3(iray.direction), s2) / divisor;
    if (b2 < 0.0f || b1 + b2 > 1.0f)
        return -1.0f;

    float t = dot(e2, s2) / divisor;
    if (t < 0.0f)
        return -1.0f;

    return 1;
}

__device__ bool ray_aabb_intersect (ray iray, aabb box) {
    float3 invDir = 1.0f / make_float3(iray.direction);
    float3 tbot = invDir * (make_float3(box.upper) - make_float3(iray.origin));
    float3 ttop = invDir * (make_float3(box.lower) - make_float3(iray.origin));

    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), tmax.z);

    return smallest_tmax > largest_tmin;
}

__device__ float ray_aabb_intersect_p (ray iray, aabb box) {
    float3 invDir = 1.0f / make_float3(iray.direction);
    float3 tbot = invDir * (make_float3(box.upper) - make_float3(iray.origin));
    float3 ttop = invDir * (make_float3(box.lower) - make_float3(iray.origin));

    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), tmax.z);

    return smallest_tmax > largest_tmin ? largest_tmin : -1.0f;
}

// closest hit program
__device__ intersection closestHitProgram(const ray &iray, const bvh_tree<int3, float4, true> &tree)
{
    intersection isect;
    isect.distance = -1.0f;

    // traverse the tree
    int stack[64];
    int *stackPtr = stack;
    *stackPtr++ = 0;

    const bvh_node *nodes = tree.nodes;
    const int3 * objects = tree.objects;
    const aabb *aabbs = tree.aabbs;
    const float4 *buffer = tree.buffer;

    do {
        const int idx = *--stackPtr;
        const auto lidx = nodes[idx].left_idx;
        const auto ridx = nodes[idx].right_idx;
        if (ray_aabb_intersect(iray, aabbs[lidx])) {
            const auto obj_idx = nodes[lidx].object_idx;
            if (obj_idx != 0xFFFFFFFF) { // leaf nodes
                const auto obj = objects[obj_idx];
                // const auto t = intersectTriangle(iray, obj, buffer);
                const auto t = ray_aabb_intersect_p(iray, aabbs[lidx]);
                if ( (isect.distance < 0.0f || t < isect.distance)) {
                    isect.distance = t;
                    // isect.index = obj_idx;
                }
            } else { // internal node
                *stackPtr++ = lidx;
            }
        }
        if (ray_aabb_intersect(iray, aabbs[ridx])) {
            const auto obj_idx = nodes[ridx].object_idx;
            if (obj_idx != 0xFFFFFFFF) { // leaf nodes
                const auto obj = objects[obj_idx];
                // const auto t = intersectTriangle(iray, obj, buffer);
                const auto t = ray_aabb_intersect_p(iray, aabbs[ridx]);
                if ( (isect.distance < 0.0f || t < isect.distance)) {
                    isect.distance = t;
                    // isect.index = obj_idx;
                }
            } else { // internal node
                *stackPtr++ = ridx;
            }
        }
    } while (stack != stackPtr);

    return isect;
}
template <typename Index, typename BufferType>
__global__ void traceRay(cudaSurfaceObject_t surf, int width, int height, const camera *cam, const bvh_tree<Index, BufferType, true> *tree)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > width || y > height)
        return;

    // generate ray
    float u = x / (float)width;
    float v = y / (float)height;
    float2 uv = make_float2(u, v);
    float2 jitter = make_float2(0.0f, 0.0f);
    ray mray = getCamRay(cam, uv, jitter);
    
    float depth = 1e20f;

    // ray-triangle intersection
    intersection isect = closestHitProgram(mray, *tree);
    // if (isect.distance > 0.0f) {
        depth = isect.distance;
    // }

    float far = 10.0f;
    float near = 0.1f;

    depth = (depth - near) / (far - near);

    surf2Dwrite(make_float4(depth, depth, depth, 1.0f), surf, x * sizeof(float4), y, cudaBoundaryModeZero);
}

void render_frame(cudaSurfaceObject_t surf, int width, int height, const bvh_tree<int3, float4, true> &tree) {
    camera *cam_d = nullptr;
    cudaMalloc(&cam_d, sizeof(camera));
    // generate camera
    float fov = 45.0f;
    camera cam{
        make_float4(0, 0, -2000, 0),
        make_float4(0, 0, 1, 0),
        make_float4(0, 1, 0, 0),
        make_float2(fov, fov * height / width),
        0.01f,
        0.1f,
    };
    cudaMemcpy(cam_d, &cam, sizeof(camera), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    bvh_tree<int3, float4, true> *d_tree;
    cudaMalloc(&d_tree, sizeof(bvh_tree<int3, float4, true>));
    cudaMemcpy(d_tree, &tree, sizeof(bvh_tree<int3, float4, true>), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    traceRay<<<grid, block>>>(surf, width, height, cam_d, d_tree);

    cudaFree(cam_d);
    cudaFree(d_tree);
}