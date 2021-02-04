#include "Math.cuh"

namespace Math {
    template <typename T>
    struct Vec4 {
        T x, y, z, w;
    
        template <typename _H = T, typename _V = T, typename _K = T, typename _Q = T>
        __host__ __device__ inline Vec4(const _H& x = 0, const _V& y = 0, const _K& z = 0, const _Q& w = 0) noexcept {
            this->x = static_cast<T>(x); this->y = static_cast<T>(y); this->z = static_cast<T>(z); this->w = static_cast<T>(w);
        }
    
        __host__ __device__ inline void Clamp(const float min, const float max) noexcept {
            using ::Utility::Clamp;
            this->x = Clamp(this->x, min, max); this->y = Clamp(this->y, min, max); this->z = Clamp(this->z, min, max); this->w = Clamp(this->w, min, max);
        }
    
        __host__ __device__ inline float GetLength3D() const noexcept { return sqrtf(this->x * this->x + this->y * this->y + this->z * this->z); }
        __host__ __device__ inline float GetLength4D() const noexcept { return sqrtf(this->x * this->x + this->y * this->y + this->z * this->z + this->w * this->w); }
    
        __host__ __device__ inline void Normalize3D() noexcept { const float l = this->GetLength3D(); this->operator/=(Vec4f32(l, l, l, 1.f)); }
        __host__ __device__ inline void Normalize4D() noexcept { this->operator/=(this->GetLength4D()); }
    
        template <typename _U>
        __host__ __device__ inline Vec4<T>& operator+=(const Vec4<_U>& o) noexcept {
            this->x += o.x; this->y += o.y; this->z += o.z; this->w += o.w;
            return *this;
        }
    
        template <typename _U>
        __host__ __device__ inline Vec4<T>& operator-=(const Vec4<_U>& o) noexcept {
            this->x -= o.x; this->y -= o.y; this->z -= o.z; this->w -= o.w;
            return *this;
        }
    
        template <typename _U>
        __host__ __device__ inline Vec4<T>& operator*=(const Vec4<_U>& o) noexcept {
            this->x *= o.x; this->y *= o.y; this->z *= o.z; this->w *= o.w;
            return *this;
        }
    
        template <typename _U>
        __host__ __device__ inline Vec4<T>& operator/=(const Vec4<_U>& o) noexcept {
            this->x /= o.x; this->y /= o.y; this->z /= o.z; this->w /= o.w;
            return *this;
        }
    
    
        template <typename _U>
        __host__ __device__ inline Vec4<T>& operator+=(const _U& n) noexcept {
            this->x += n; this->y += n; this->z += n; this->w += n;
            return *this;
        }
    
        template <typename _U>
        __host__ __device__ inline Vec4<T>& operator-=(const _U& n) noexcept {
            this->x -= n; this->y -= n; this->z -= n; this->w -= n;
            return *this;
        }
    
        template <typename _U>
        __host__ __device__ inline Vec4<T>& operator*=(const _U& n) noexcept {
            this->x *= n; this->y *= n; this->z *= n; this->w *= n;
            return *this;
        }
    
        template <typename _U>
        __host__ __device__ inline Vec4<T>& operator/=(const _U& n) noexcept {
            this->x /= n; this->y /= n; this->z /= n; this->w /= n;
            return *this;
        }
    }; // class Vec4<T>
    
    template <typename T, typename _U>
    __host__ __device__ inline auto operator+(const Vec4<T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x + b.x)> {
        return Vec4<decltype(a.x + b.x)>{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
    }
    
    template <typename T, typename _U>
    __host__ __device__ inline auto operator-(const Vec4<T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x - b.x)> {
        return Vec4<decltype(a.x - b.x)>{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
    }
    
    template <typename T, typename _U>
    __host__ __device__ inline auto operator*(const Vec4<T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x* b.x)> {
        return Vec4<decltype(a.x* b.x)>{ a.x* b.x, a.y* b.y, a.z* b.z, a.w* b.w };
    }
    
    template <typename T, typename _U>
    __host__ __device__ inline auto operator/(const Vec4<T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x / b.x)> {
        return Vec4<decltype(a.x / b.x)>{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
    }
    
    
    template <typename T, typename _U>
    __host__ __device__ inline auto operator+(const Vec4<T>& a, const _U& n) noexcept -> Vec4<decltype(a.x + n)> {
        return Vec4<decltype(a.x + n)>{ a.x + n, a.y + n, a.z + n, a.w + n };
    }
    
    template <typename T, typename _U>
    __host__ __device__ inline auto operator-(const Vec4<T>& a, const _U& n) noexcept -> Vec4<decltype(a.x - n)> {
        return Vec4<decltype(a.x - n)>{ a.x - n, a.y - n, a.z - n, a.w - n };
    }
    
    template <typename T, typename _U>
    __host__ __device__ inline auto operator*(const Vec4<T>& a, const _U& n) noexcept -> Vec4<decltype(a.x* n)> {
        return Vec4<decltype(a.x* n)>{ a.x* n, a.y* n, a.z* n, a.w* n };
    }
    
    template <typename T, typename _U>
    __host__ __device__ inline auto operator/(const Vec4<T>& a, const _U& n) noexcept -> Vec4<decltype(a.x / n)> {
        return Vec4<decltype(a.x / n)>{ a.x / n, a.y / n, a.z / n, a.w / n };
    }
    
    
    template <typename T, typename _U>
    __host__ __device__ inline auto operator+(const _U& n, const Vec4<T>& a) { return a * n; }
    
    template <typename T, typename _U>
    __host__ __device__ inline auto operator-(const _U& n, const Vec4<T>& a) { return a * n; }
    
    template <typename T, typename _U>
    __host__ __device__ inline auto operator*(const _U& n, const Vec4<T>& a) { return a * n; }
    
    template <typename T, typename _U>
    __host__ __device__ inline auto operator/(const _U& n, const Vec4<T>& a) { return a * n; }
    

    template <typename T, typename U>
    __host__ __device__ inline decltype(std::declval<T>() + std::declval<U>()) DotProduct3D(const Vec4<T> a, const Vec4<U> b) noexcept {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    template <typename T, typename U>
    __host__ __device__ inline decltype(std::declval<T>() + std::declval<U>()) DotProduct4D(const Vec4<T> a, const Vec4<U> b) noexcept{
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    template <typename T>
    __host__ __device__ inline Vec4<T> Normalized3D(Vec4<T> v) noexcept {
        v.Normalize3D();
        return v;
    }

    template <typename T>
    __host__ __device__ inline Vec4<T> Normalized4D(Vec4<T> v) noexcept {
        v.Normalize4D();
        return v;
    }

    template <typename T, typename U>
    __host__ __device__ inline Vec4<decltype(std::declval<T>() + std::declval<U>())> Reflected3D(const Vec4<T> inDirection, const Vec4<U> normal) noexcept {
        return inDirection - 2 * DotProduct3D(inDirection, normal) * normal;
    }

    template <typename T, typename U>
    __host__ __device__ inline Vec4<decltype(std::declval<T>() + std::declval<U>())> CrossProduct3D(const Vec4<T> a, const Vec4<U> b) noexcept {
        return Vec4<decltype(a.x + b.x)>{
            a.y*b.z-a.z*b.y,
            a.z*b.x-a.x*b.z,
            a.x*b.y-a.y*b.x,
            0
        };
    }

    template <typename T>
    __host__ __device__ inline Vec4<T> Clamped(Vec4<T> v, const float min, const float max) noexcept {
        v.Clamp(min, max);
        return v;
    }

    template <typename T>
    __device__ inline Vec4<T> Random3DUnitVector(curandState_t* const pRandSate) noexcept {
        return Normalized3D(Vec4<T>(2.0f * Utility::RandomFloat(pRandSate) - 1.0f,
                                    2.0f * Utility::RandomFloat(pRandSate) - 1.0f,
                                    2.0f * Utility::RandomFloat(pRandSate) - 1.0f,
                                    0.f));
    }

    template <typename T, typename U>
    __host__ __device__ inline Vec4<decltype(std::declval<T>() + std::declval<U>())> Refracted(const Vec4<T> in, Vec4<U> n, const float ior) noexcept {
        auto cosi = ::Utility::Clamp(DotProduct3D(in, n), -1.f, 1.f); 
        float etai = 1.f, etat = ior; 
        if (cosi < 0) {
            cosi = -cosi;
        } else {
            ::Utility::Swap(etai, etat);
            n *= -1;
        }
    
        auto eta = etai / etat; 
        auto k = 1 - eta * eta * (1 - cosi * cosi); 
    
        return (k < 0) ? Vec4<decltype(std::declval<T>() + std::declval<U>())>(0, 0, 0, 0) : (eta * in + (eta * cosi - sqrtf(k)) * n); 
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& stream, const Vec4<T>& v) noexcept {
        stream << '(' << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ')';
    
        return stream;
    }
}; // namespace Math