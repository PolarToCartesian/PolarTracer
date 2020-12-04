#ifndef __POLARTRACER__FILE_MATH_CUH
#define __POLARTRACER__FILE_MATH_CUH

#include "Pch.cuh"
#include "Common.cuh"

__constant__ const float  EPSILON = 0.0001f;
__constant__ const size_t MAX_REC = 5u;
__constant__ const size_t SPP     = 5u;

namespace Math {
    template <typename T> struct Vec4;

    typedef Vec4<std::uint8_t> Coloru8;
    typedef Vec4<float>        Colorf32;
    typedef Vec4<float>        Vec4f32;

    template <typename T> std::ostream& operator<<(std::ostream& stream, const Vec4<T>& v) noexcept;

    template <typename T, typename U> __host__ __device__ inline decltype(std::declval<T>() + std::declval<U>()) DotProduct3D(const Vec4<T> a, const Vec4<U> b) noexcept;
    template <typename T, typename U> __host__ __device__ inline decltype(std::declval<T>() + std::declval<U>()) DotProduct4D(const Vec4<T> a, const Vec4<U> b) noexcept;

    template <typename T> __host__ __device__ inline Vec4<T> Normalized3D(Vec4<T> v) noexcept; 
    template <typename T> __host__ __device__ inline Vec4<T> Normalized4D(Vec4<T> v) noexcept;

    template <typename T, typename U> __host__ __device__ inline Vec4<decltype(std::declval<T>() + std::declval<U>())> Reflected3D   (const Vec4<T> inDirection, const Vec4<U> normal)             noexcept;
    template <typename T, typename U> __host__ __device__ inline Vec4<decltype(std::declval<T>() + std::declval<U>())> CrossProduct3D(const Vec4<T> a,           const Vec4<U> b)                  noexcept;
    template <typename T, typename U> __host__ __device__ inline Vec4<decltype(std::declval<T>() + std::declval<U>())> Refracted     (const Vec4<T> in,                Vec4<U> n, const float ior) noexcept;

    template <typename T> __host__ __device__ inline Vec4<T> Clamped(Vec4<T> v, const T min, const T max) noexcept;
}; // namespace Math

#define blackf32       Math::Colorf32{0.f, 0.f, 0.f, 1.f}
#define whitef32       Math::Colorf32{1.f, 1.f, 1.f, 1.f}
#define redf32         Math::Colorf32{1.f, 0.f, 0.f, 1.f}
#define greenf32       Math::Colorf32{0.f, 1.f, 0.f, 1.f}
#define bluef32        Math::Colorf32{0.f, 0.f, 1.f, 1.f}
#define transparentf32 Math::Colorf32{0.f, 0.f, 0.f, 0.f}

#define blacku8       Math::Coloru8{0u, 0u, 0u, 255u}
#define whiteu8       Math::Coloru8{1u, 1u, 1u, 255u}
#define redu8         Math::Coloru8{1u, 0u, 0u, 255u}
#define greenu8       Math::Coloru8{0u, 1u, 0u, 255u}
#define blueu8        Math::Coloru8{0u, 0u, 1u, 255u}
#define transparentu8 Math::Coloru8{255u, 255u, 255u, 255u}

#endif // __POLARTRACER__FILE_MATH_CUH