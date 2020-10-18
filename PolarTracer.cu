#include <cmath>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <string>
#include <memory>
#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>

#ifdef _WIN32
    #include <wrl.h>
    #include <wincodec.h>

    #define THROW_FATAL_ERROR(msg) MessageBoxA(NULL, msg, "Error", MB_ICONERROR)
#else
    #define THROW_FATAL_ERROR(msg) std::cerr << msg << '\n'
#endif

struct Coloru8 {
    std::uint8_t r, g, b, a;
}; // Coloru8

struct Colorf32 {
    float r, g, b, a;
}; // Colorf32

// Constants
class Image {
private:
    const std::uint16_t m_width;
    const std::uint16_t m_height;
    const std::uint32_t m_nPixels;

    std::unique_ptr<Coloru8[]> m_pBuff;

public:
    Image() = default;

    Image(const std::uint16_t width, const std::uint16_t height) noexcept 
      : m_width(width), m_height(height), m_nPixels(static_cast<std::uint32_t>(width) * height)
    {
        this->m_pBuff = std::make_unique<Coloru8[]>(this->m_nPixels);
    }

    inline std::uint16_t GetWidth()      const noexcept { return this->m_width;   }
    inline std::uint16_t GetHeight()     const noexcept { return this->m_height;  }
    inline std::uint32_t GetPixelCount() const noexcept { return this->m_nPixels; }

    inline Coloru8* GetBufferPtr() const noexcept { return this->m_pBuff.get(); }

    inline       Coloru8& operator()(const size_t i)       noexcept { return this->m_pBuff[i]; }
    inline const Coloru8& operator()(const size_t i) const noexcept { return this->m_pBuff[i]; }

    inline       Coloru8& operator()(const size_t x, const size_t y)       noexcept { return this->m_pBuff[y * this->m_width + this->m_height]; }
    inline const Coloru8& operator()(const size_t x, const size_t y) const noexcept { return this->m_pBuff[y * this->m_width + this->m_height]; }

    void Save(const std::string& filename) noexcept {
#ifdef _WIN32 // Use WIC (Windows Imaging Component) To Save A .PNG File Natively

      const std::string  fullFilename = filename + ".png";
      const std::wstring fullFilenameW(fullFilename.begin(), fullFilename.end());
  
      Microsoft::WRL::ComPtr<IWICImagingFactory>    factory;
      Microsoft::WRL::ComPtr<IWICBitmapEncoder>     bitmapEncoder;
      Microsoft::WRL::ComPtr<IWICBitmapFrameEncode> bitmapFrame;
      Microsoft::WRL::ComPtr<IWICStream>            outputStream;
  
      if (CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)) != S_OK)
          THROW_FATAL_ERROR("[WIC] Could Not Create IWICImagingFactory");
  
      if (factory->CreateStream(&outputStream) != S_OK)
          THROW_FATAL_ERROR("[WIC] Failed To Create Output Stream");
  
      if (outputStream->InitializeFromFilename(fullFilenameW.c_str(), GENERIC_WRITE) != S_OK)
          THROW_FATAL_ERROR("[WIC] Failed To Initialize Output Stream From Filename");
  
      if (factory->CreateEncoder(GUID_ContainerFormatPng, NULL, &bitmapEncoder) != S_OK)
          THROW_FATAL_ERROR("[WIC] Failed To Create Bitmap Encoder");
  
      if (bitmapEncoder->Initialize(outputStream.Get(), WICBitmapEncoderNoCache) != S_OK)
          THROW_FATAL_ERROR("[WIC] Failed To Initialize Bitmap ");
  
      if (bitmapEncoder->CreateNewFrame(&bitmapFrame, NULL) != S_OK)
          THROW_FATAL_ERROR("[WIC] Failed To Create A New Frame");
  
      if (bitmapFrame->Initialize(NULL) != S_OK)
          THROW_FATAL_ERROR("[WIC] Failed To Initialize A Bitmap's Frame");
  
      if (bitmapFrame->SetSize(this->m_width, this->m_height) != S_OK)
          THROW_FATAL_ERROR("[WIC] Failed To Set A Bitmap's Frame's Size");
  
      const WICPixelFormatGUID desiredPixelFormat = GUID_WICPixelFormat32bppBGRA;
  
      WICPixelFormatGUID currentPixelFormat = {};
      if (bitmapFrame->SetPixelFormat(&currentPixelFormat) != S_OK)
          THROW_FATAL_ERROR("[WIC] Failed To Set Pixel Format On A Bitmap Frame's");
  
      if (!IsEqualGUID(currentPixelFormat, desiredPixelFormat))
          THROW_FATAL_ERROR("[WIC] The Requested Pixel Format Is Not Supported");
  
      if (bitmapFrame->WritePixels(this->m_height, this->m_width * sizeof(Coloru8), this->m_nPixels * sizeof(Coloru8), (BYTE*)this->m_pBuff.get()) != S_OK)
          THROW_FATAL_ERROR("[WIC] Failed To Write Pixels To A Bitmap's Frame");
  
      if (bitmapFrame->Commit() != S_OK)
          THROW_FATAL_ERROR("[WIC] Failed To Commit A Bitmap's Frame");
  
      if (bitmapEncoder->Commit() != S_OK)
          THROW_FATAL_ERROR("[WIC] Failed To Commit Bitmap Encoder");
  
#else // On Other Operating Systems, Simply Write A .PAM File
  
      const std::string fullFilename = filename + ".pam";
  
      // Open
      FILE* fp = std::fopen(fullFilename.c_str(), "wb");
  
      if (fp) {
          // Header
          std::fprintf(fp, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\n MAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", this->m_width, this->m_height);
  
          // Write Contents
          std::fwrite(this->m_pBuff.get(), this->m_nPixels * sizeof(Coloru8), 1u, fp);
  
          // Close
          std::fclose(fp);
      }
#endif
    }
}; // Image

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

  // Allocate 1 thread per pixel of coordinates (X,Y). Use as many blocks in the grid as needed
  // The RayTrace function will use the thread's index (both in the grid and in a block) to determine the pixel it will trace rays through
  const dim3 dimBlock = dim3(32, 32); // 32 warps of 32 threads per block
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

int main(int argc, char** argv) {
  Image image(1920, 1080);
  RayTraceScene(image);
  image.Save("frame");
}
