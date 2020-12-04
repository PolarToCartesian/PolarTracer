#ifndef __POLARTRACER__FILE_CAMERA_CUH
#define __POLARTRACER__FILE_CAMERA_CUH

#include "Math.cuh"
#include "Memory.cuh"

struct Camera {
    float         fov;
    Math::Vec4f32 position;

    __device__ inline Ray GenerateCameraRay(const size_t pixelX, const size_t pixelY, const size_t surfaceW, const size_t surfaceH, curandState_t* const randState) noexcept {
        Ray ray;
        ray.origin = this->position;
        ray.direction = Math::Normalized3D(Math::Vec4f32(
            (2.0f  * ((pixelX + Utility::RandomFloat(randState)) / static_cast<float>(surfaceW)) - 1.0f) * tan(this->fov) * static_cast<float>(surfaceW) / static_cast<float>(surfaceH),
            (-2.0f * ((pixelY + Utility::RandomFloat(randState)) / static_cast<float>(surfaceH)) + 1.0f) * tan(this->fov),
            1.0f,
            0.f));

        return ray;
    }
}; // Camera

#endif // __POLARTRACER__FILE_CAMERA_CUH