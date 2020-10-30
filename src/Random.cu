#ifndef __POLAR_TRACER__RANDOM_CU
#define __POLAR_TRACER__RANDOM_CU

#include <curand.h>
#include <curand_kernel.h>

#include "Enums.cu"
#include "Vector.cu"

__device__ float RandomFloat(curandState_t* const randState) noexcept {
    return curand_uniform(randState);
}

__device__ inline Vec4f32 Random3DUnitVector(curandState_t* const randState) noexcept {
    return Vec4f32::Normalized3D(Vec4f32(2.0f * RandomFloat(randState) - 1.0f,
        2.0f * RandomFloat(randState) - 1.0f,
        2.0f * RandomFloat(randState) - 1.0f,
        0.f));
}

#endif // __POLAR_TRACER__RANDOM_CU