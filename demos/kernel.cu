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

__global__ void kernel(float4 *framebuffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int index = y * width + x;
    framebuffer[index] = make_float4(x / (float)width, y / (float)height, 0.2f, 1.0f);
}

__global__ void kernelSurf(cudaSurfaceObject_t surf, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    surf2Dwrite(make_float4(x / (float)width, y / (float)height, 0.2f, 1.0f), surf, x * sizeof(float4), y, cudaBoundaryModeZero);
}

void renderer(float4 *framebuffer, int width, int height) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    kernel<<<grid, block>>>(framebuffer, width, height);
}

void renderSurf(cudaSurfaceObject_t surf, int width, int height) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    kernelSurf<<<grid, block>>>(surf, width, height);
}

