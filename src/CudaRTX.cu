// LIBC
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <cstdio>
#include <cassert>
#include <cstdint>
#include <utility>
#include <iostream>
#include <type_traits>

#define MAX_DEPTH 5

namespace PRTX {

    enum class Device { CPU = 10, GPU = 100 }; // Device

    template <typename _T, ::PRTX::Device _D>
    class Pointer {
    private:
        _T* m_raw = nullptr;
        
    public:
        __host__ __device__ inline Pointer() noexcept {  }

        template <typename _U>
        __host__ __device__ inline Pointer(_U* const p)              noexcept : m_raw(p)       { static_assert(std::is_same<_T, _U>::value); }
        __host__ __device__ inline Pointer(const Pointer<_T, _D>& o) noexcept : m_raw(o.m_raw) {  }

        __host__ __device__ inline _T* Get() const noexcept { return this->m_raw; }

        template <typename _U = _T>
        __host__ __device__ inline void operator=(_U* const p)                      noexcept { this->m_raw = p;       static_assert(std::is_same<_T, _U>::value); }
        __host__ __device__ inline void operator=(const ::PRTX::Pointer<_T, _D>& o) noexcept { this->m_raw = o.m_raw; }

        __host__ __device__ inline operator _T*&      ()       noexcept { return this->m_raw; }
        __host__ __device__ inline operator _T* const&() const noexcept { return this->m_raw; }

        __host__ __device__ inline _T* operator->() const noexcept { return this->m_raw; }
    }; // Pointer<_T>

    template <typename _T>
    using CPU_PTR = ::PRTX::Pointer<_T, ::PRTX::Device::CPU>;

    template <typename _T>
    using GPU_PTR = ::PRTX::Pointer<_T, ::PRTX::Device::GPU>;

    template <typename _T, ::PRTX::Device _D>
    __host__ __device__ inline ::PRTX::Pointer<_T, _D> AllocateSize(const size_t size) noexcept {
        if constexpr (_D == ::PRTX::Device::CPU) {
            return ::PRTX::CPU_PTR<_T>(reinterpret_cast<_T*>(std::malloc(size)));
        } else {
            _T* p;
            cudaMalloc(&p, size);
            return ::PRTX::GPU_PTR<_T>(p);
        }
    }

    template <typename _T, ::PRTX::Device _D>
    __host__ __device__ inline ::PRTX::Pointer<_T, _D> AllocateCount(const size_t count) noexcept {
        return ::PRTX::AllocateSize<_T, _D>(count * sizeof(_T));
    }

    template <typename _T, ::PRTX::Device _D>
    __host__ __device__ inline void Free(const ::PRTX::Pointer<_T, _D>& p) noexcept {
        if constexpr (_D == ::PRTX::Device::CPU) {
            std::free(p);
        } else {
            cudaFree(reinterpret_cast<void*>(p.Get()));
        }
    }

    template <typename _T, ::PRTX::Device _D_DST, ::PRTX::Device _D_SRC>
    __host__ __device__ inline void CopySize(const ::PRTX::Pointer<_T, _D_DST>& dst, const ::PRTX::Pointer<_T, _D_SRC>& src, const size_t size) noexcept {
        if constexpr (_D_SRC == ::PRTX::Device::CPU && _D_DST == ::PRTX::Device::CPU) {
            cudaMemcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyHostToHost);
        } else if constexpr (_D_SRC == ::PRTX::Device::GPU && _D_DST == ::PRTX::Device::GPU) {
            cudaMemcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        } else if constexpr (_D_SRC == ::PRTX::Device::CPU && _D_DST == ::PRTX::Device::GPU) {
            cudaMemcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
        } else if constexpr (_D_SRC == ::PRTX::Device::GPU && _D_DST == ::PRTX::Device::CPU) {
            cudaMemcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        } else { static_assert(1 == 1, "Incompatible Destination and Source Arguments"); }
    }

    template <typename _T, ::PRTX::Device _D_DST, ::PRTX::Device _D_SRC>
    __host__ __device__ inline void CopyCount(const ::PRTX::Pointer<_T, _D_DST>& dst, const ::PRTX::Pointer<_T, _D_SRC>& src, const size_t count) noexcept {
        ::PRTX::CopySize(dst, src, count * sizeof(_T));
    }

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
        Colorf32 emittance;
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

        ::PRTX::CPU_PTR<Coloru8> m_pBuff;

    public:
        Image() = default;

        Image(const std::uint16_t width, const std::uint16_t height) noexcept 
        : m_width(width), m_height(height), m_nPixels(static_cast<std::uint32_t>(width) * height)
        {
            this->m_pBuff = ::PRTX::AllocateSize<Coloru8, ::PRTX::Device::CPU>(this->m_nPixels * sizeof(Coloru8));
        }

        ~Image() {
            ::PRTX::Free(this->m_pBuff);
        }

        inline std::uint16_t GetWidth()      const noexcept { return this->m_width;   }
        inline std::uint16_t GetHeight()     const noexcept { return this->m_height;  }
        inline std::uint32_t GetPixelCount() const noexcept { return this->m_nPixels; }

        inline ::PRTX::CPU_PTR<Coloru8> GetBufferPtr() const noexcept { return this->m_pBuff; }

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
                std::fwrite(this->m_pBuff, this->m_nPixels * sizeof(Coloru8), 1u, fp);
    
                // Close
                std::fclose(fp);
            }
        }

    }; // Image

    namespace details {

        struct PolarTracerNonMeshData {
            size_t width  = 0;
            size_t height = 0;

            Camera camera{};
            
            // constants computed with the "camera" object
            float cameraProjectW = 0;
            float cameraProjectH = 0;
        };

        template <typename _T>
        __device__ inline _T Clamp(const _T& x, const _T& min, const _T& max) noexcept { return min(max(x, min), max); }

        template <size_t _N>
        __device__ Colorf32 RayTrace(const Ray& ray) {
            const size_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

            return Colorf32(pixelX / (float)1920, pixelY / (float)1080, 0.0f, 1.0f);
        }

        __device__ __host__ inline float RandomFloat() noexcept {
            return 0.5f;
        }

        __device__ inline Ray GenerateCameraRay(const size_t& pixelX, const size_t& pixelY, const ::PRTX::GPU_PTR<PolarTracerNonMeshData>& pDeviceNMD) noexcept {
            Ray ray;
            ray.origin    = Vec4f32(0.f, 0.f, 0.f, 0.f);
            ray.direction = Vec4f32(
                (2.0f  * ((pixelX + RandomFloat()) / float(pDeviceNMD->width)  - 1.0f) * pDeviceNMD->cameraProjectW),
                (-2.0f * ((pixelY + RandomFloat()) / float(pDeviceNMD->height) + 1.0f) * pDeviceNMD->cameraProjectH),
                1.0f,
                0.f);
          
            return ray;
        }

        // Can't pass arguments via const& because these variables exist on the host and not on the device
        __global__ void RayTracingDispatcher(const PRTX::GPU_PTR<Coloru8> pSurface, const PRTX::GPU_PTR<PolarTracerNonMeshData> pDeviceNMD) {
            // Calculate the thread's (X, Y) location
            const size_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

            // Bounds check
            if (pixelX >= pDeviceNMD->width || pixelY >= pDeviceNMD->height) return;

            // Determine the pixel's index into the image buffer
            const size_t index = pixelX + pixelY * pDeviceNMD->width;

            const Ray cameraRay = GenerateCameraRay(pixelX, pixelY, pDeviceNMD);

            // the current pixel's color (represented with floating point components)
            Colorf32 pixelColor = RayTrace<0>(cameraRay) * 255.f;

            // Save the result to the buffer
            *(pSurface.Get() + index) = Coloru8(pixelColor.x, pixelColor.y, pixelColor.z, pixelColor.w);
        }

    }; // details

    class PolarTracer {
    private:
        struct {
            ::PRTX::details::PolarTracerNonMeshData m_nonMeshData;
        } host;

        struct {
            PRTX::GPU_PTR<Coloru8> m_pRenderBuffer;
            PRTX::GPU_PTR<::PRTX::details::PolarTracerNonMeshData> m_pNonMeshData;
        } device;
    
    public:
        PolarTracer(const size_t width, const size_t height, const Camera& camera, const std::vector<Sphere>& spheres) {
            this->host.m_nonMeshData.width  = width;
            this->host.m_nonMeshData.height = height;
            this->host.m_nonMeshData.camera = camera;
            this->host.m_nonMeshData.cameraProjectH = std::tan(camera.fov);
            this->host.m_nonMeshData.cameraProjectW = this->host.m_nonMeshData.cameraProjectH * width / height;

            this->device.m_pRenderBuffer = PRTX::AllocateSize<Coloru8, PRTX::Device::GPU>(width * height);
            this->device.m_pNonMeshData  = PRTX::AllocateSize<::PRTX::details::PolarTracerNonMeshData, PRTX::Device::GPU>(1);

            const auto src = ::PRTX::CPU_PTR<::PRTX::details::PolarTracerNonMeshData>(&this->host.m_nonMeshData);
            ::PRTX::CopySize(this->device.m_pNonMeshData, src, sizeof(::PRTX::details::PolarTracerNonMeshData));
        }

        inline void RayTraceScene(const Image& outSurface) {
            assert(outSurface.GetWidth() == this->host.m_nonMeshData.width && outSurface.GetHeight() == this->host.m_nonMeshData.height);

            const size_t bufferSize = outSurface.GetPixelCount() * sizeof(Coloru8);
    
            // Allocate 1 thread per pixel of coordinates (X,Y). Use as many blocks in the grid as needed
            // The RayTrace function will use the thread's index (both in the grid and in a block) to determine the pixel it will trace rays through
            const dim3 dimBlock = dim3(32, 32); // 32 warps of 32 threads per block (=1024 threads in total which is the hardware limit)
            const dim3 dimGrid  = dim3(std::ceil(this->host.m_nonMeshData.width  / static_cast<float>(dimBlock.x)),
                                       std::ceil(this->host.m_nonMeshData.height / static_cast<float>(dimBlock.y)));
    
            // trace rays through each pixel
            ::PRTX::details::RayTracingDispatcher<<<dimGrid, dimBlock>>>(this->device.m_pRenderBuffer, this->device.m_pNonMeshData);
        
            // wait for the job to finish
            cudaDeviceSynchronize();
    
            // copy the gpu buffer to a new cpu buffer
            ::PRTX::CopySize(outSurface.GetBufferPtr(), this->device.m_pRenderBuffer, bufferSize);
        }

        inline ~PolarTracer() {
            ::PRTX::Free(this->device.m_pRenderBuffer);
            ::PRTX::Free(this->device.m_pNonMeshData);
        }
    }; // PolarTracer

}; // PRTX

int main(int argc, char** argv) {
    const size_t WIDTH = 1920, HEIGHT = 1080;
    PRTX::Image image(WIDTH, HEIGHT);

    PRTX::Camera camera;
    camera.position = PRTX::Vec4f32();
    camera.fov      = M_PI / 4.f;

    std::vector<PRTX::Sphere> spheres(1);
    spheres[0].position = PRTX::Vec4f32{0.0f, 0.0f, 2.f, 0.f};
    spheres[0].radius   = 0.25f;
    spheres[0].material.diffuse   = PRTX::Colorf32{1.f, 0.f, 1.f, 1.f};
    spheres[0].material.emittance = PRTX::Colorf32{0.f, 0.f, 0.f, 0.f};
    
    PRTX::PolarTracer pt(WIDTH, HEIGHT, camera, spheres);
    pt.RayTraceScene(image);

    image.Save("frame");
}