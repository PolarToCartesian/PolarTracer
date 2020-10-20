#ifndef __PRTX__MATH_CU
#define __PRTX__MATH_CU

#include <cstdint>
#include <ostream>

namespace PRTX {

    template <typename _T>
    struct Vec4 {
        _T x, y, z, w;

        template <typename _H = _T, typename _V = _T, typename _K = _T, typename _Q = _T>
        __host__ __device__ inline Vec4(const _H& x = 0, const _V& y = 0, const _K& z = 0, const _Q& w = 0) noexcept {
            this->x = static_cast<_T>(x); this->y = static_cast<_T>(y); this->z = static_cast<_T>(z); this->w = static_cast<_T>(w);
        }

        template <typename _U>
        __host__ __device__ inline Vec4<_T>& operator+=(const Vec4<_U>& o) noexcept {
          this->x += o.x; this->y += o.y; this->z += o.z; this->w += o.w;
          return *this; 
        }

        template <typename _U>
        __host__ __device__ inline Vec4<_T>& operator-=(const Vec4<_U>& o) noexcept {
          this->x -= o.x; this->y -= o.y; this->z -= o.z; this->w -= o.w;
          return *this;
        }

        template <typename _U>
        __host__ __device__ inline Vec4<_T>& operator*=(const Vec4<_U>& o) noexcept {
          this->x *= o.x; this->y *= o.y; this->z *= o.z; this->w *= o.w;
          return *this;
        }

        template <typename _U>
        __host__ __device__ inline Vec4<_T>& operator/=(const Vec4<_U>& o) noexcept {
          this->x /= o.x; this->y /= o.y; this->z /= o.z; this->w /= o.w;
          return *this;
        }


        template <typename _U>
        __host__ __device__ inline Vec4<_T>& operator+=(const _U& n) noexcept {
          this->x += n; this->y += n; this->z += n; this->w += n;
          return *this;
        }

        template <typename _U>
        __host__ __device__ inline Vec4<_T>& operator-=(const _U& n) noexcept {
          this->x -= n; this->y -= n; this->z -= n; this->w -= n;
          return *this;
        }

        template <typename _U>
        __host__ __device__ inline Vec4<_T>& operator*=(const _U& n) noexcept {
          this->x *= n; this->y *= n; this->z *= n; this->w *= n;
          return *this;
        }

        template <typename _U>
        __host__ __device__ inline Vec4<_T>& operator/=(const _U& n) noexcept {
          this->x /= n; this->y /= n; this->z /= n; this->w /= n;
          return *this;
        }
    }; // Vec4<_T>

    template <typename _T, typename _U>
    __host__ __device__ inline auto operator+(const Vec4<_T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x + b.x)> {
      return Vec4<decltype(a.x + b.x)>{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; 
    }
    
    template <typename _T, typename _U>
    __host__ __device__ inline auto operator-(const Vec4<_T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x - b.x)> {
      return Vec4<decltype(a.x - b.x)>{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
    }

    template <typename _T, typename _U>
    __host__ __device__ inline auto operator*(const Vec4<_T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x * b.x)> {
      return Vec4<decltype(a.x * b.x)>{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
    }

    template <typename _T, typename _U>
    __host__ __device__ inline auto operator/(const Vec4<_T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x / b.x)> {
      return Vec4<decltype(a.x / b.x)>{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
    }


    template <typename _T, typename _U>
    __host__ __device__ inline auto operator+(const Vec4<_T>& a, const _U& n) noexcept -> Vec4<decltype(a.x + n)> {
      return Vec4<decltype(a.x + n)>{ a.x + n, a.y + n, a.z + n, a.w + n };
    }

    template <typename _T, typename _U>
    __host__ __device__ inline auto operator-(const Vec4<_T>& a, const _U& n) noexcept -> Vec4<decltype(a.x - n)> {
      return Vec4<decltype(a.x - n)>{ a.x - n, a.y - n, a.z - n, a.w - n };
    }

    template <typename _T, typename _U>
    __host__ __device__ inline auto operator*(const Vec4<_T>& a, const _U& n) noexcept -> Vec4<decltype(a.x * n)> {
      return Vec4<decltype(a.x * n)>{ a.x * n, a.y * n, a.z * n, a.w * n };
    }

    template <typename _T, typename _U>
    __host__ __device__ inline auto operator/(const Vec4<_T>& a, const _U& n) noexcept -> Vec4<decltype(a.x / n)> {
      return Vec4<decltype(a.x / n)>{ a.x / n, a.y / n, a.z / n, a.w / n };
    }


    template <typename _T, typename _U>
    __host__ __device__ inline auto operator+(const _U& n, const Vec4<_T>& a) { return a * n; }

    template <typename _T, typename _U>
    __host__ __device__ inline auto operator-(const _U& n, const Vec4<_T>& a) { return a * n; }

    template <typename _T, typename _U>
    __host__ __device__ inline auto operator*(const _U& n, const Vec4<_T>& a) { return a * n; }

    template <typename _T, typename _U>
    __host__ __device__ inline auto operator/(const _U& n, const Vec4<_T>& a) { return a * n; }

    template <typename _T>
    std::ostream& operator<<(std::ostream& stream, const Vec4<_T>& v) noexcept {
        stream << '(' << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ')';

        return stream;
    }

    typedef ::PRTX::Vec4<std::uint8_t> Coloru8;
    typedef ::PRTX::Vec4<float>        Colorf32;
    typedef ::PRTX::Vec4<float>        Vec4f32;

    struct Ray {
        ::PRTX::Vec4f32 origin;
        ::PRTX::Vec4f32 direction;
    }; // Ray

}; // PRTX

#endif // __PRTX__MATH_CU