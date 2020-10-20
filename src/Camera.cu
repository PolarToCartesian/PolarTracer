#ifndef __PRTX__CAMERA_CU
#define __PRTX__CAMERA_CU

#include "Math.cu"
#include "Misc.cu"
#include "Memory.cu"
#include "Random.cu"

namespace PRTX {

    struct Camera {
        float   fov;
        ::PRTX::Vec4f32 position;
    }; // Camera

    struct RenderParams {
        size_t width  = 0;
        size_t height = 0;

        ::PRTX::Camera camera;
    };

    __device__ inline Ray GenerateCameraRay(const size_t& pixelX, const size_t& pixelY, const ::PRTX::GPU_ptr<::PRTX::RenderParams>& pRanderParams) noexcept {
        const ::PRTX::RenderParams& renderParams = *pRanderParams;

        Ray ray;
        ray.origin    = Vec4f32(0.f, 0.f, 0.f, 0.f);
        ray.direction = Vec4f32(
            2.0f  * ((pixelX + RandomFloat()) / float(renderParams.width)  - 1.0f) * tan(renderParams.camera.fov) * renderParams.width / renderParams.height,
            -2.0f * ((pixelY + RandomFloat()) / float(renderParams.height) + 1.0f) * tan(renderParams.camera.fov),
            1.0f,
            0.f);
      
        return ray;
    }

}; // namespace PRTX

#endif // __PRTX__CAMERA_CU