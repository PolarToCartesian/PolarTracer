#ifndef __POLAR_TRACER__UTIL_CU
#define __POLAR_TRACER__UTIL_CU

template <typename T>
__host__ __device__ inline void Swap(T& a, T& b) noexcept {
    const T& tmp = a;
    a = b;
    b = tmp;
}

template <typename T>
__host__ __device__ inline T Clamp(const T& x, const T& min, const T& max) {
    return x <= min ? min : (x >= max ? max : x);
}

#endif // __POLAR_TRACER__UTIL_CU