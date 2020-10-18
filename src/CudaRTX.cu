#include "Common.hpp"

#include <thrust/device_vector.h>

// Small helper functions
template <typename _T>
__device__ _T Clamp(const _T& x, const _T& min, const _T& max) noexcept {
    if (x > max) return max;
    if (x < min) return min;

    return x;
}

// float4 vector operator overloading!
__device__ inline float4 operator+(const float4& a, const float4& b) noexcept { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ inline float4 operator-(const float4& a, const float4& b) noexcept { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__device__ inline float4 operator*(const float4& a, const float4& b) noexcept { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ inline float4 operator/(const float4& a, const float4& b) noexcept { return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }

template <typename T> __device__ inline float4 operator+(const float4& a, const T& n) noexcept { return make_float4(a.x + n, a.y + n, a.z + n, a.w + n); }
template <typename T> __device__ inline float4 operator-(const float4& a, const T& n) noexcept { return make_float4(a.x - n, a.y - n, a.z - n, a.w - n); }
template <typename T> __device__ inline float4 operator*(const float4& a, const T& n) noexcept { return make_float4(a.x * n, a.y * n, a.z * n, a.w * n); }
template <typename T> __device__ inline float4 operator/(const float4& a, const T& n) noexcept { return make_float4(a.x / n, a.y / n, a.z / n, a.w / n); }

__global__ void RayTrace(std::uint8_t* pBuffer, const size_t width, const size_t height) {
    // Calculate the thread's (X, Y) location
    const size_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

    // Bounds check
    if (pixelX >= width || pixelY >= height) return;

    // Determine the pixel's index into the image buffer
    const size_t index = pixelX + pixelY * width;
    
    // the current pixel's color (represented with floating point components)
    float4 pixelColor = make_float4(pixelX / (float)width, pixelY / (float)height, 0.0f, 1.0f);
    
    // Convert the color values from float4s to uchar4s
    pixelColor = make_float4(255.f, 255.f, 255.f, 255.f) * pixelColor;
    uchar4 uchar4FinalColor = make_uchar4(pixelColor.x, pixelColor.y, pixelColor.z, pixelColor.w);

    // Save the result to the buffer
    *(reinterpret_cast<uchar4*>(pBuffer) + index) = uchar4FinalColor;
}

void RayTraceScene(Image& outSurface) {
    const size_t bufferSize = outSurface.GetPixelCount() * sizeof(Coloru8);

    // Allocate memory on the GPU for the render surface    
    std::uint8_t* gpuBuffer;
    cudaMallocManaged(&gpuBuffer, bufferSize);

    // Allocate 1 thread per pixel of coordinates (X,Y). Use as many blocks in the grid as needed
    // The RayTrace function will use the thread's index (both in the grid and in a block) to determine the pixel it will trace rays through
    const dim3 dimBlock = dim3(32, 32); // 32 warps of 32 threads per block
    const dim3 dimGrid  = dim3(std::ceil(outSurface.GetWidth()  / static_cast<float>(dimBlock.x)),
                               std::ceil(outSurface.GetHeight() / static_cast<float>(dimBlock.y)));

    // trace rays through each pixel
    RayTrace<<<dimGrid, dimBlock>>>(gpuBuffer, outSurface.GetWidth(), outSurface.GetHeight());
  
    // wait for the job to finish
    cudaDeviceSynchronize();

    // copy the gpu buffer to a new cpu buffer
    cudaMemcpy(outSurface.GetBufferPtr(), gpuBuffer, bufferSize, cudaMemcpyDeviceToHost);

    // free the GPU image buffer
    cudaFree(gpuBuffer);
}