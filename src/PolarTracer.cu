#ifndef __PRTX__POLAR_TRACER_CU
#define __PRTX__POLAR_TRACER_CU

#include "Misc.cu"
#include "Math.cu"
#include "Image.cu"
#include "Memory.cu"
#include "Camera.cu"
#include "Objects.cu"

#include <cassert>

namespace PRTX {

    namespace details {

        template <size_t _N>
        __device__ Colorf32 RayTrace(const Ray& ray) {
            const size_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

            return Colorf32(pixelX / (float)1920, pixelY / (float)1080, 0.0f, 1.0f);
        }

        // Can't pass arguments via const& because these variables exist on the host and not on the device
        __global__ void RayTracingDispatcher(const ::PRTX::Pointer<Coloru8, ::PRTX::Device::GPU> pSurface, const PRTX::GPU_ptr<::PRTX::RenderParams> pParams) {
            // Calculate the thread's (X, Y) location
            const size_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
            const size_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

            // Bounds check
            if (pixelX >= pParams->width || pixelY >= pParams->height) return;

            // Determine the pixel's index into the image buffer
            const size_t index = pixelX + pixelY * pParams->width;

            const Ray cameraRay = GenerateCameraRay(pixelX, pixelY, pParams);

            // the current pixel's color (represented with floating point components)
            Colorf32 pixelColor = RayTrace<0>(cameraRay) * 255.f;

            // Save the result to the buffer
            *(pSurface + index) = Coloru8(pixelColor.x, pixelColor.y, pixelColor.z, pixelColor.w);
        }

    }; // details

    class PolarTracer {
    private:
        struct {
            ::PRTX::RenderParams m_renderParams;
        } host;

        struct {
            ::PRTX::Image<Coloru8, ::PRTX::Device::GPU> m_frameBuffer;
            ::PRTX::GPU_ptr<::PRTX::RenderParams> m_pRenderParams;
        } device;
    
    public:
        PolarTracer(const ::PRTX::RenderParams& renderParams, const ::PRTX::CPU_Array<Sphere>& spheres)
            : host{renderParams}
        {
            this->device.m_frameBuffer   = ::PRTX::Image<Coloru8, ::PRTX::Device::GPU>(renderParams.width, renderParams.height);
            this->device.m_pRenderParams = ::PRTX::AllocateSingle<::PRTX::RenderParams, PRTX::Device::GPU>();

            const auto src = ::PRTX::CPU_ptr<::PRTX::RenderParams>(&this->host.m_renderParams);
            ::PRTX::CopySingle(this->device.m_pRenderParams, src);
        }

        inline void RayTraceScene(const ::PRTX::Image<::PRTX::Coloru8, ::PRTX::Device::CPU>& outSurface) {
            assert(outSurface.GetWidth() == this->host.m_renderParams.width && outSurface.GetHeight() == this->host.m_renderParams.height);

            const size_t bufferSize = outSurface.GetPixelCount() * sizeof(Coloru8);
    
            // Allocate 1 thread per pixel of coordinates (X,Y). Use as many blocks in the grid as needed
            // The RayTrace function will use the thread's index (both in the grid and in a block) to determine the pixel it will trace rays through
            const dim3 dimBlock = dim3(32, 32); // 32 warps of 32 threads per block (=1024 threads in total which is the hardware limit)
            const dim3 dimGrid  = dim3(std::ceil(this->host.m_renderParams.width  / static_cast<float>(dimBlock.x)),
                                       std::ceil(this->host.m_renderParams.height / static_cast<float>(dimBlock.y)));
    
            // trace rays through each pixel
            ::PRTX::details::RayTracingDispatcher<<<dimGrid, dimBlock>>>(this->device.m_frameBuffer.GetData(), this->device.m_pRenderParams);
        
            // wait for the job to finish
            cudaDeviceSynchronize();
    
            // copy the gpu buffer to a new cpu buffer
            ::PRTX::CopySize(outSurface.GetData(), this->device.m_frameBuffer.GetData(), bufferSize);
        }

        inline ~PolarTracer() {
            ::PRTX::Free(this->device.m_pRenderParams);
        }
    }; // PolarTracer

}; // namespace PRTX

#endif // __PRTX__POLAR_TRACER_CU