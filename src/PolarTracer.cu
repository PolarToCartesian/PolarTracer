#ifndef __PRTX__POLAR_TRACER_CU
#define __PRTX__POLAR_TRACER_CU

#include "Math.cu"
#include "Image.cu"
#include "Memory.cu"
#include "Camera.cu"
#include "Objects.cu"

#include <cassert>

namespace PRTX {

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

        __device__ inline Ray GenerateCameraRay(const size_t& pixelX, const size_t& pixelY, const ::PRTX::GPU_ptr<PolarTracerNonMeshData>& pDeviceNMD) noexcept {
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
        __global__ void RayTracingDispatcher(const PRTX::GPU_ptr<Coloru8> pSurface, const PRTX::GPU_ptr<PolarTracerNonMeshData> pDeviceNMD) {
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
            *(pSurface + index) = Coloru8(pixelColor.x, pixelColor.y, pixelColor.z, pixelColor.w);
        }

    }; // details

    class PolarTracer {
    private:
        struct {
            ::PRTX::details::PolarTracerNonMeshData m_nonMeshData;
        } host;

        struct {
            PRTX::GPU_Array<Coloru8> m_pRenderBuffer;
            PRTX::GPU_ptr<::PRTX::details::PolarTracerNonMeshData> m_pNonMeshData;
        } device;
    
    public:
        PolarTracer(const size_t width, const size_t height, const Camera& camera, const ::PRTX::CPU_Array<Sphere>& spheres) {
            this->host.m_nonMeshData.width  = width;
            this->host.m_nonMeshData.height = height;
            this->host.m_nonMeshData.camera = camera;
            this->host.m_nonMeshData.cameraProjectH = std::tan(camera.fov);
            this->host.m_nonMeshData.cameraProjectW = this->host.m_nonMeshData.cameraProjectH * width / height;

            this->device.m_pRenderBuffer = PRTX::GPU_Array<Coloru8>(width * height);
            this->device.m_pNonMeshData  = PRTX::AllocateCount<::PRTX::details::PolarTracerNonMeshData, PRTX::Device::GPU>(1);

            const auto src = ::PRTX::CPU_ptr<::PRTX::details::PolarTracerNonMeshData>(&this->host.m_nonMeshData);
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
            ::PRTX::CopySize(outSurface.GetBufferPtr(), this->device.m_pRenderBuffer.GetData(), bufferSize);
        }

        inline ~PolarTracer() {
            ::PRTX::Free(this->device.m_pNonMeshData);
        }
    }; // PolarTracer

}; // namespace PRTX

#endif // __PRTX__POLAR_TRACER_CU