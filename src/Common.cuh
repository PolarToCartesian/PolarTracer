#ifndef __POLARTRACER__FILE_COMMON_CUH
#define __POLARTRACER__FILE_COMMON_CUH

// The "Device" enum class represents the devices from which
// memory can be accessed. This is necessary because the cpu can't
// read/write directly from/to the GPU's memory and conversely.
enum class Device { CPU, GPU }; // Device

// +-----------+
// | Utilities |
// +-----------+

namespace Utility {
    template <typename T>
    __host__ __device__ inline void Swap(T& a, T& b) noexcept {
        const T& tmp = a; a = b; b = tmp;
    }

    template <typename T>
    __host__ __device__ inline T Clamp(const T& x, const T& min, const T& max) noexcept {
        return x <= min ? min : (x >= max ? max : x);
    }

    __device__ inline float RandomFloat(curandState_t* randState) noexcept {
        return curand_uniform(randState);
    }
}; // namespace Utility

#endif // __POLARTRACER__FILE_COMMON_CUH