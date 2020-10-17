#include <cmath>
#include <memory>
#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>

#include "Image.hpp"

// Small helper functions
template <typename _T>
inline _T Clamp(const _T& x, const _T& min, const _T& max) noexcept {
  if (x > max) return max;
  if (x < min) return min;

  return x;
}

__global__ void RayTrace(Colorf32* pFloatImage, const size_t width, const size_t height) {
  // Calculate the thread's (X, Y) location
  const size_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

  // Bounds check
  if (pixelX >= width || pixelY >= height) return;

  // Determine the pixel's index into the image buffer
  const size_t index = pixelX + pixelY * width;
  

  
  *(pFloatImage + index) = Colorf32{pixelX / (float)width, pixelY/(float)height, 0.5f, 1.f};
}

void RayTraceScene(Image& cpuOutputImage) {
  const size_t f32BufferSize = cpuOutputImage.GetPixelCount() * sizeof(Colorf32);

  // allocate raw GPU Buffer
  float* gpuImageBuffer;
  cudaMallocManaged(&gpuImageBuffer, f32BufferSize);

  // calculate the __
  const size_t nThreadsPerBlock = 512; // 32*32*32 (32*32 warps of 32 threads)
  const size_t nBlocks          = std::ceil(cpuOutputImage.GetPixelCount() / static_cast<float>(nThreadsPerBlock));

  const dim3 dimBlock = dim3(16, 16);
  const dim3 dimGrid  = dim3(std::ceil(cpuOutputImage.GetWidth()  / static_cast<float>(dimBlock.x)),
                             std::ceil(cpuOutputImage.GetHeight() / static_cast<float>(dimBlock.y)));

  // trace rays through each pixel
  RayTrace<<<dimGrid, dimBlock>>>((Colorf32*)gpuImageBuffer, cpuOutputImage.GetWidth(), cpuOutputImage.GetHeight());
  
  // wait for the job to finish
  cudaDeviceSynchronize();

  // copy the gpu buffer to a new cpu buffer
  std::unique_ptr<float[]> cpuF32Buffer = std::make_unique<float[]>(cpuOutputImage.GetPixelCount() * 4);
  cudaMemcpy(cpuF32Buffer.get(), gpuImageBuffer, f32BufferSize, cudaMemcpyDeviceToHost);

  // free the GPU image buffer
  cudaFree(gpuImageBuffer);

  // write the image data to the image
  for (std::uint32_t i = 0; i < cpuOutputImage.GetPixelCount(); i++) {
    Coloru8& pixel = cpuOutputImage(i);
    pixel.r = static_cast<std::uint8_t>(Clamp(cpuF32Buffer[i * 4 + 0] * 255.f, 0.f, 255.f));
    pixel.g = static_cast<std::uint8_t>(Clamp(cpuF32Buffer[i * 4 + 1] * 255.f, 0.f, 255.f));
    pixel.b = static_cast<std::uint8_t>(Clamp(cpuF32Buffer[i * 4 + 2] * 255.f, 0.f, 255.f));
    pixel.a = static_cast<std::uint8_t>(Clamp(cpuF32Buffer[i * 4 + 3] * 255.f, 0.f, 255.f));
  }
}