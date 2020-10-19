// LIBC
#include <cmath>
#include <memory>
#include <string>
#include <cstdio>
#include <cstdint>
#include <utility>
#include <iostream>

// CUDA
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>

#define MAX_DEPTH 5

namespace PolarTracer {

    template <typename _T>
    struct Vec4 {
        _T x, y, z, w;

        template <typename _H = _T, typename _V = _T, typename _K = _T, typename _Q = _T>
        __host__ __device__ inline Vec4(const _H& x = 0, const _V& y = 0, const _K& z = 0, const _Q& w = 0) noexcept {
            this->x = static_cast<_T>(x); this->y = static_cast<_T>(y); this->z = static_cast<_T>(z); this->w = static_cast<_T>(w);
        }

        template <typename _U> __host__ __device__ inline Vec4<_T>& operator+=(const Vec4<_U>& o) noexcept { this->x += o.x; this->y += o.y; this->z += o.z; this->w += o.w; return *this; }
        template <typename _U> __host__ __device__ inline Vec4<_T>& operator-=(const Vec4<_U>& o) noexcept { this->x -= o.x; this->y -= o.y; this->z -= o.z; this->w -= o.w; return *this; }
        template <typename _U> __host__ __device__ inline Vec4<_T>& operator*=(const Vec4<_U>& o) noexcept { this->x *= o.x; this->y *= o.y; this->z *= o.z; this->w *= o.w; return *this; }
        template <typename _U> __host__ __device__ inline Vec4<_T>& operator/=(const Vec4<_U>& o) noexcept { this->x /= o.x; this->y /= o.y; this->z /= o.z; this->w /= o.w; return *this; }

        template <typename _U> __host__ __device__ inline Vec4<_T>& operator+=(const _U& n) noexcept { this->x += n; this->y += n; this->z += n; this->w += n; return *this; }
        template <typename _U> __host__ __device__ inline Vec4<_T>& operator-=(const _U& n) noexcept { this->x -= n; this->y -= n; this->z -= n; this->w -= n; return *this; }
        template <typename _U> __host__ __device__ inline Vec4<_T>& operator*=(const _U& n) noexcept { this->x *= n; this->y *= n; this->z *= n; this->w *= n; return *this; }
        template <typename _U> __host__ __device__ inline Vec4<_T>& operator/=(const _U& n) noexcept { this->x /= n; this->y /= n; this->z /= n; this->w /= n; return *this; }
    }; // Vec4<_T>

    template <typename _T, typename _U> __host__ __device__ inline auto operator+(const Vec4<_T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x + b.x)> { return Vec4<decltype(a.x + b.x)>{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
    template <typename _T, typename _U> __host__ __device__ inline auto operator-(const Vec4<_T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x - b.x)> { return Vec4<decltype(a.x - b.x)>{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
    template <typename _T, typename _U> __host__ __device__ inline auto operator*(const Vec4<_T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x * b.x)> { return Vec4<decltype(a.x * b.x)>{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w }; }
    template <typename _T, typename _U> __host__ __device__ inline auto operator/(const Vec4<_T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x / b.x)> { return Vec4<decltype(a.x / b.x)>{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w }; }

    template <typename _T, typename _U> __host__ __device__ inline auto operator+(const Vec4<_T>& a, const _U& n) noexcept -> Vec4<decltype(a.x + n)> { return Vec4<decltype(a.x + n)>{ a.x + n, a.y + n, a.z + n, a.w + n }; }
    template <typename _T, typename _U> __host__ __device__ inline auto operator-(const Vec4<_T>& a, const _U& n) noexcept -> Vec4<decltype(a.x - n)> { return Vec4<decltype(a.x - n)>{ a.x - n, a.y - n, a.z - n, a.w - n }; }
    template <typename _T, typename _U> __host__ __device__ inline auto operator*(const Vec4<_T>& a, const _U& n) noexcept -> Vec4<decltype(a.x * n)> { return Vec4<decltype(a.x * n)>{ a.x * n, a.y * n, a.z * n, a.w * n }; }
    template <typename _T, typename _U> __host__ __device__ inline auto operator/(const Vec4<_T>& a, const _U& n) noexcept -> Vec4<decltype(a.x / n)> { return Vec4<decltype(a.x / n)>{ a.x / n, a.y / n, a.z / n, a.w / n }; }

    template <typename _T>
    std::ostream& operator<<(std::ostream& stream, const Vec4<_T>& v) noexcept {
        stream << '(' << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ')';

        return stream;
    }

    typedef Vec4<std::uint8_t> Coloru8;
    typedef Vec4<float>        Colorf32;
    typedef Vec4<float>        Vec4f32;

    struct Material {
        Colorf32 diffuse;
    }; // Material

    struct Sphere {
        Vec4f32 position;
        float   radius;

        Material material;
    }; // Sphere

    struct Ray {
        Vec4f32 origin;
        Vec4f32 direction;
    }; // Ray

    struct Camera {
        float   fov;
        Vec4f32 position;
    }; // Camera

    class Image {
    private:
        const std::uint16_t m_width   = 0;
        const std::uint16_t m_height  = 0;
        const std::uint32_t m_nPixels = 0;

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
            const std::string fullFilename = filename + ".pam";
    
            // Open
            FILE* fp = std::fopen(fullFilename.c_str(), "wb");
    
            if (fp) {
                // Header
                std::fprintf(fp, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", this->m_width, this->m_height);
    
                // Write Contents
                std::fwrite(this->m_pBuff.get(), this->m_nPixels * sizeof(Coloru8), 1u, fp);
    
                // Close
                std::fclose(fp);
            }
        }

    }; // Image

    namespace details {

        template <typename _T>
        __device__ inline _T Clamp(const _T& x, const _T& min, const _T& max) noexcept { return min(max(x, min), max); }

        template <size_t _N>
        __device__ Colorf32 RayTrace(const Ray& ray) {
            const size_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

            return Colorf32(pixelX / (float)1920, pixelY / (float)1080, 0.0f, 1.0f);
        }

        __global__ void RayTracingDispatcher(const thrust::device_ptr<std::uint8_t> pSurface, const size_t width, const size_t height) {
            // Calculate the thread's (X, Y) location
            const size_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

            // Bounds check
            if (pixelX >= width || pixelY >= height) return;

            // Determine the pixel's index into the image buffer
            const size_t index = pixelX + pixelY * width;

            Ray cameraRay{}; //...

            // the current pixel's color (represented with floating point components)
            Colorf32 pixelColor = RayTrace<0>(cameraRay) * 255.f;

            // Save the result to the buffer
            *(reinterpret_cast<Coloru8*>(thrust::raw_pointer_cast(pSurface)) + index) = Coloru8(pixelColor.x, pixelColor.y, pixelColor.z, pixelColor.w);
        }

    }; // details

    void RayTraceScene(const Image& outSurface) noexcept {
        const size_t bufferSize = outSurface.GetPixelCount() * sizeof(Coloru8);

        thrust::device_ptr<std::uint8_t> gpuBuffer = thrust::device_malloc<std::uint8_t>(bufferSize);

        // Allocate 1 thread per pixel of coordinates (X,Y). Use as many blocks in the grid as needed
        // The RayTrace function will use the thread's index (both in the grid and in a block) to determine the pixel it will trace rays through
        const dim3 dimBlock = dim3(32, 32); // 32 warps of 32 threads per block (=1024 threads in total which is the hardware limit)
        const dim3 dimGrid  = dim3(std::ceil(outSurface.GetWidth()  / static_cast<float>(dimBlock.x)),
                                std::ceil(outSurface.GetHeight() / static_cast<float>(dimBlock.y)));

        // trace rays through each pixel
        PolarTracer::details::RayTracingDispatcher<<<dimGrid, dimBlock>>>(gpuBuffer, outSurface.GetWidth(), outSurface.GetHeight());
    
        // wait for the job to finish
        cudaDeviceSynchronize();

        // copy the gpu buffer to a new cpu buffer
        cudaMemcpy(outSurface.GetBufferPtr(), thrust::raw_pointer_cast(gpuBuffer), bufferSize, cudaMemcpyDeviceToHost);

        thrust::device_free(gpuBuffer);
    }

}; // PolarTracer

int main(int argc, char** argv) {
    PolarTracer::Image image(1920, 1080);
    PolarTracer::RayTraceScene(image);
    image.Save("frame");
}