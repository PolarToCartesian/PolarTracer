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

    template <size_t _N>
    __device__ Colorf32 RayTrace(const Ray& ray, const PRTX::GPU_Ptr<::PRTX::RenderParams> pParams) {
        bool bIntersected = false;

        if (bIntersected) {
            return Colorf32(0.0f, 0.0f, 0.0f, 1.0f);
        } else {
            const float ratio = (threadIdx.y + blockIdx.y * blockDim.y) / float(pParams->height);

            const auto skyLightBlue = ::PRTX::Vec4f32(0.78f, 0.96f, 1.00f, 1.0f);
            const auto skyDarkBlue  = ::PRTX::Vec4f32(0.01f, 0.84f, 0.93f, 1.0f);

            return ::PRTX::Vec4f32(skyLightBlue * ratio + skyDarkBlue * (1 - ratio));
        }
    }

    // Can't pass arguments via const& because these variables exist on the host and not on the device
    __global__ void RayTracingDispatcher(const ::PRTX::GPU_Ptr<Coloru8> pSurface,
                                         const ::PRTX::GPU_Ptr<::PRTX::RenderParams> pParams) {
        // Calculate the thread's (X, Y) location
        const size_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

        // Bounds check
        if (pixelX >= pParams->width || pixelY >= pParams->height) return;

        // Determine the pixel's index into the image buffer
        const size_t index = pixelX + pixelY * pParams->width;

        const Ray cameraRay = GenerateCameraRay(pixelX, pixelY, pParams);

        // the current pixel's color (represented with floating point components)
        Colorf32 pixelColor = RayTrace<0>(cameraRay, pParams) * 255.f;

        // Save the result to the buffer
        *(pSurface + index) = Coloru8(pixelColor.x, pixelColor.y, pixelColor.z, pixelColor.w);
    }

    class PolarTracer {
    private:
        struct {
            ::PRTX::RenderParams m_renderParams;
        } host;

        struct {
            ::PRTX::Image<Coloru8, ::PRTX::Device::GPU> m_frameBuffer;
            ::PRTX::GPU_Ptr<::PRTX::RenderParams> m_pRenderParams;
            ::PRTX::GPU_Array<::PRTX::Sphere>     m_spheres;
        } device;
    
    public:
        PolarTracer(const ::PRTX::RenderParams& renderParams, const ::PRTX::CPU_Array<Sphere>& spheres)
            : host{renderParams}
        {
            this->device.m_frameBuffer   = ::PRTX::Image<Coloru8, ::PRTX::Device::GPU>(renderParams.width, renderParams.height);
            this->device.m_pRenderParams = ::PRTX::AllocateSingle<::PRTX::RenderParams, PRTX::Device::GPU>();
            this->device.m_spheres       = ::PRTX::GPU_Array<::PRTX::Sphere>(spheres);

            const auto src = ::PRTX::CPU_Ptr<::PRTX::RenderParams>(&this->host.m_renderParams);
            ::PRTX::CopySingle(this->device.m_pRenderParams, src);
        }

        inline void RayTraceScene(const ::PRTX::Image<::PRTX::Coloru8, ::PRTX::Device::CPU>& outSurface) {
            assert(outSurface.GetWidth() == this->host.m_renderParams.width && outSurface.GetHeight() == this->host.m_renderParams.height);

            const size_t bufferSize = outSurface.GetPixelCount() * sizeof(::PRTX::Coloru8);
    
            // Allocate 1 thread per pixel of coordinates (X,Y). Use as many blocks in the grid as needed
            // The RayTrace function will use the thread's index (both in the grid and in a block) to determine the pixel it will trace rays through
            const dim3 dimBlock = dim3(32, 32); // 32 warps of 32 threads per block (=1024 threads in total which is the hardware limit)
            const dim3 dimGrid  = dim3(std::ceil(this->host.m_renderParams.width  / static_cast<float>(dimBlock.x)),
                                       std::ceil(this->host.m_renderParams.height / static_cast<float>(dimBlock.y)));
    
            // trace rays through each pixel
            ::PRTX::RayTracingDispatcher<<<dimGrid, dimBlock>>>(this->device.m_frameBuffer.GetData(), this->device.m_pRenderParams);
        
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