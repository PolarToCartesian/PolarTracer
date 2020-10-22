#include <limits>
#include <ostream>
#include <cassert>
#include <optional>

#define EPSILON (float)0.1f
#define FLT_MAX (float)std::numeric_limits<float>::max()
#define MAX_REC (10)

__device__ size_t randomNumberIndex = 0u;
__device__ float  randomNumbers[100] = { 0.199597f, 0.604987f, 0.255558f, 0.421514f, 0.720092f, 0.815522f, 0.192279f, 0.385067f, 0.350586f, 0.397595f, 0.357564f, 0.748578f, 0.00414681f, 0.533777f, 0.995393f, 0.907929f, 0.494525f, 0.472084f, 0.864498f, 0.695326f, 0.938409f, 0.785484f, 0.290453f, 0.13312f, 0.943201f, 0.926033f, 0.320409f, 0.0662487f, 0.25414f, 0.421945f, 0.667499f, 0.444524f, 0.838885f, 0.908202f, 0.8063f, 0.291879f, 0.114376f, 0.875398f, 0.247916f, 0.045868f, 0.535327f, 0.491882f, 0.642606f, 0.184197f, 0.154249f, 0.14628f, 0.939923f, 0.979867f, 0.503506f, 0.478285f, 0.491597f, 0.0545161f, 0.847528f, 0.0108021f, 0.934526f, 0.282655f, 0.0207591f, 0.329495f, 0.328761f, 0.560112f, 0.119835f, 0.296947f, 0.289384f, 0.83466f, 0.164883f, 0.0987901f, 0.0792031f, 0.258547f, 0.0754077f, 0.0143626f, 0.318207f, 0.483693f, 0.0715536f, 0.998425f, 0.322974f, 0.879418f, 0.261024f, 0.49866f, 0.453179f, 0.347203f, 0.638452f, 0.274543f, 0.595394f, 0.640481f, 0.798533f, 0.680735f, 0.95186f, 0.4518f, 0.969803f, 0.419822f, 0.00485671f, 0.727772f, 0.475605f, 0.816288f, 0.55194f, 0.550753f, 0.601672f, 0.908048f, 0.35448f, 0.863961f };

__device__ float RandomFloat() noexcept {
    return randomNumbers[(randomNumberIndex++) % (sizeof(randomNumbers) / sizeof(float))];
}

template <typename T>
__host__ __device__ inline T Clamp(const T& x, const T& min, const T& max) noexcept {
    return (x > max) ? max : ((x < min) ? min : x);
}

// The "Device" enum class represents the devices from which
// memory can be accessed. This is necessary because the cpu can't
// read/write directly from/to the GPU's memory and conversely.
enum class Device { CPU, GPU }; // Device

// The Pointer<typename T, Device D> class represents a C++ Pointer of base type
// T that is accessible from the device D (view enum Device).
template <typename T, Device D>
class Pointer {
private:
    T* m_raw = nullptr;
    
public:
    __host__ __device__ inline Pointer() noexcept {  }

    __host__ __device__ inline Pointer(T* const p)              noexcept { this->SetPointer(p); }
    __host__ __device__ inline Pointer(const Pointer<T, D>& o) noexcept { this->SetPointer(o); }

    __host__ __device__ inline void SetPointer(T* const p)              noexcept { this->m_raw = p; }
    __host__ __device__ inline void SetPointer(const Pointer<T, D>& o) noexcept { this->SetPointer(o.m_raw); }

    __host__ __device__ inline T*&       GetPointer()       noexcept { return this->m_raw; }
    __host__ __device__ inline T* const& GetPointer() const noexcept { return this->m_raw; }

    template <typename U>
    __host__ __device__ inline Pointer<U, D> AsPointerTo() const noexcept {
        return Pointer<U, D>(reinterpret_cast<U*>(this->m_raw));
    }

    __host__ __device__ inline void operator=(T* const p)                      noexcept { this->SetPointer(p); }
    __host__ __device__ inline void operator=(const Pointer<T, D>& o) noexcept { this->SetPointer(o); }

    __host__ __device__ inline operator T*&      ()       noexcept { return this->m_raw; }
    __host__ __device__ inline operator T* const&() const noexcept { return this->m_raw; }

    __host__ __device__ inline std::conditional_t<std::is_same_v<T, void>, int, T>& operator[](const size_t i) noexcept {
        static_assert(!std::is_same_v<T, void>, "Can't Index A Pointer To A Void");
        return *(this->m_ptr + i);
    }
    
    __host__ __device__ inline const std::conditional_t<std::is_same_v<T, void>, int, T>& operator[](const size_t i) const noexcept {
        static_assert(!std::is_same_v<T, void>, "Can't Index A Pointer To A Void");
        return *(this->m_ptr + i);
    }

    __host__ __device__ inline T* const operator->() const noexcept { return this->m_raw; }
}; // Pointer<T>

// Some aliases for the Pointer<T, D> class.
template <typename T>
using CPU_Ptr = Pointer<T, Device::CPU>;

template <typename T>
using GPU_Ptr = Pointer<T, Device::GPU>;

// Memory Allocation
template <typename T, Device D>
__host__ __device__ inline Pointer<T, D> AllocateSize(const size_t size) noexcept {
    if constexpr (D == Device::CPU) {
        return CPU_Ptr<T>(reinterpret_cast<T*>(std::malloc(size)));
    } else {
        T* p;
        cudaMalloc(reinterpret_cast<void**>(&p), size);
        return GPU_Ptr<T>(p);
    }
}

template <typename T, Device D>
__host__ __device__ inline Pointer<T, D> AllocateCount(const size_t count) noexcept {
    return AllocateSize<T, D>(count * sizeof(T));
}

template <typename T, Device D>
__host__ __device__ inline Pointer<T, D> AllocateSingle() noexcept {
    return AllocateSize<T, D>(sizeof(T));
}

// Memory Deallocation

template <typename T, Device D>
__host__ __device__ inline void Free(const Pointer<T, D>& p) noexcept {
    if constexpr (D == Device::CPU) {
        std::free(p);
    } else {
        cudaFree(p.template AsPointerTo<void>());
    }
}

// Copying Memory
// These functions take as arguments objects of type _PTR<T, D>
// such as Pointer<T, D>, ArrayView<T, D> or Array<T, D>.
// The object _PTR<T, D> has to be convertible to a raw pointer of
// base type T accessible by the device D.

template <template<typename, Device> typename _PTR_DST, typename T_DST, Device D_DST,
          template<typename, Device> typename _PTR_SRC, typename T_SRC, Device D_SRC>
__host__ __device__ inline void CopySize(const _PTR_DST<T_DST, D_DST>& dst,
                                            const _PTR_SRC<T_SRC, D_SRC>& src,
                                            const size_t size) noexcept
{
    static_assert(std::is_same_v<T_DST, T_SRC>, "Incompatible Source And Destination Raw Pointer Types");
    
    cudaMemcpyKind memcpyKind;

    if constexpr (D_SRC == Device::CPU && D_DST == Device::CPU) {
        memcpyKind = cudaMemcpyKind::cudaMemcpyHostToHost;
    } else if constexpr (D_SRC == Device::GPU && D_DST == Device::GPU) {
        memcpyKind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
    } else if constexpr (D_SRC == Device::CPU && D_DST == Device::GPU) {
        memcpyKind = cudaMemcpyKind::cudaMemcpyHostToDevice;
    } else if constexpr (D_SRC == Device::GPU && D_DST == Device::CPU) {
        memcpyKind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
    } else { static_assert(1 == 1, "Incompatible Destination and Source Arguments"); }

    cudaMemcpy(dst, src, size, memcpyKind);
}

template <template<typename, Device> typename _PTR_DST, typename T_DST, Device D_DST,
          template<typename, Device> typename _PTR_SRC, typename T_SRC, Device D_SRC>
__host__ __device__ inline void CopyCount(const _PTR_DST<T_DST, D_DST>& dst,
                                          const _PTR_SRC<T_SRC, D_SRC>& src,
                                          const size_t count) noexcept
{
    static_assert(std::is_same_v<T_DST, T_SRC>, "Incompatible Source And Destination Raw Pointer Types");

    CopySize(dst, src, count * sizeof(T_DST));
}

template <template<typename, Device> typename _PTR_DST, typename T_DST, Device D_DST,
          template<typename, Device> typename _PTR_SRC, typename T_SRC, Device D_SRC>
__host__ __device__ inline void CopySingle(const _PTR_DST<T_DST, D_DST>& dst,
                                           const _PTR_SRC<T_SRC, D_SRC>& src) noexcept
{
    static_assert(std::is_same_v<T_DST, T_SRC>, "Incompatible Source And Destination Raw Pointer Types");
    
    CopySize(dst, src, sizeof(T_DST));
}


// The UniquePointer<typename T, Device D> class represents a C++ Pointer
// of base type T whose memory is owned and managed by this class. As a result,
// when this class is destroyed or it's owning memory location changes, it will
// free the memory it owned.
template <typename T, Device D>
class UniquePointer {
private:
    Pointer<T, D> m_ptr;

public:
    __host__ __device__ inline UniquePointer() noexcept {
        this->Free();
        this->m_ptr = nullptr;
    }
    
    template <typename... _ARGS>
    __host__ __device__ inline UniquePointer(const _ARGS&... args) noexcept {
        this->Free();
        this->m_ptr = new (AllocateSingle<T, D>()) T(std::forward<_ARGS>(args)...);
    }

    __host__ __device__ inline UniquePointer(const Pointer<T, D>& o)  noexcept {
        this->Free();
        this->m_ptr = o;
    }

    __host__ __device__ inline UniquePointer(UniquePointer<T, D>&& o) noexcept {
        this->Free();
        this->m_ptr = o.m_ptr;
        o.m_ptr     = nullptr;
    }

    __host__ __device__ inline void Free() const noexcept {
        // Since we use placement new, we have to call T's destructor ourselves
        this->m_ptr->~T();

        // and then free the memory
        ::Free(this->m_ptr);
    }

    __host__ __device__ inline ~UniquePointer() noexcept { this->Free(); }

    __host__ __device__ inline UniquePointer<T, D>& operator=(const Pointer<T, D>& o)  noexcept {
        this->Free();
        this->m_ptr = o;
        return *this;
    }

    __host__ __device__ inline UniquePointer<T, D>& operator=(UniquePointer<T, D>&& o) noexcept {
        this->Free();
        this->m_ptr = o;
        o.m_ptr     = nullptr;
        return *this;
    }

    __host__ __device__ inline const Pointer<T, D>& GetPointer() const noexcept { return this->m_ptr; }

    __host__ __device__ inline operator const Pointer<T, D>&() const noexcept { return this->m_ptr; }

    __host__ __device__ inline operator T* const&() const noexcept { return this->m_ptr; }

    __host__ __device__ inline T* const operator->() const noexcept { return this->m_ptr; }

    __host__ __device__ inline       T& operator[](const size_t i)       noexcept { return *(this->m_ptr + i); }
    __host__ __device__ inline const T& operator[](const size_t i) const noexcept { return *(this->m_ptr + i); }


    __host__ __device__ UniquePointer(const UniquePointer<T, D>& o) = delete;
    __host__ __device__ UniquePointer<T, D>& operator=(const UniquePointer<T, D>& o) = delete;
}; // UniquePointer<T, D>

// Some aliases for the UniquePointer<T, D> class.
template <typename T>
using CPU_UniquePtr = UniquePointer<T, Device::CPU>;

template <typename T>
using GPU_UniquePtr = UniquePointer<T, Device::GPU>;

// The ArrayView<typename T, Device D> class represents a
// contiguous allocation of memory on the device D of elements of type
// T. It is defined by a starting memory address and a count of elements
// of type T following the address.
template <typename T, Device D>
class ArrayView {
private:
    size_t m_count = 0;
    Pointer<T, D> m_pBegin;

protected:
    __host__ __device__ inline void SetPointer(const Pointer<T, D>& pBegin) noexcept { this->m_pBegin = pBegin; }
    __host__ __device__ inline void SetCount  (const size_t count)                    noexcept { this->m_count  = count;  }

    __host__ __device__ inline operator T*& () noexcept { return this->m_pBegin; }

public:
    __host__ __device__ inline ArrayView() noexcept = default;

    __host__ __device__ inline ArrayView(const Pointer<T, D>& pBegin, const size_t count) noexcept
        : m_pBegin(pBegin), m_count(count)
    {  }

    __host__ __device__ inline const Pointer<T, D>& GetPointer() const noexcept { return this->m_pBegin; }
    __host__ __device__ inline const size_t&                  GetCount()   const noexcept { return this->m_count;  }

    __host__ __device__ inline operator const Pointer<T, D>&() const noexcept { return this->m_pBegin; }

    __host__ __device__ inline operator T* const&() const noexcept { return this->m_pBegin; }

    __host__ __device__ inline       T& operator[](const size_t i)       noexcept { return *(this->m_pBegin + i); }
    __host__ __device__ inline const T& operator[](const size_t i) const noexcept { return *(this->m_pBegin + i); }

    ~ArrayView() = default;
}; // ArrayView<T, D>

// Some aliases for the ArrayView<T, D> class.
template <typename T>
using CPU_ArrayView = ArrayView<T, Device::CPU>;

template <typename T>
using GPU_ArrayView = ArrayView<T, Device::GPU>;


// The Array<typename T, Device D> is essentialy a
// ArrayView<T, D> who owns the memory it represents.
template <typename T, Device D>
class Array : public ArrayView<T, D> {
public:
    __host__ __device__ inline Array() noexcept = default;

    __host__ __device__ inline Array(const size_t count) noexcept {
        this->SetCount(count);
        this->SetPointer(AllocateCount<T, D>(count));
    }
    
    template <Device D_O>
    __host__ __device__ inline Array(const Array<T, D_O>& o) noexcept
        : Array(o.GetCount())
    {
        CopyCount(*this, o, this->GetCount());
    }

    __host__ __device__ inline Array(Array<T, D>&& o) noexcept
    {
        this->SetCount(o.GetCount());
        o.SetCount(0);
        this->SetPointer(o.GetPointer());
        o.SetPointer((T*)nullptr);
    }

    __host__ __device__ inline Array<T, D>& operator=(Array<T, D>&& o) noexcept {
        this->SetCount(o.GetCount());
        o.SetCount(0);
        this->SetPointer(o.GetPointer());
        o.SetPointer((T*)nullptr);

        return *this;
    }

    __host__ __device__ inline void Reserve(const size_t count) noexcept {
        const auto newCount = this->GetCount() + count;
        const auto newBegin = AllocateCount<T, D>(newCount);

        CopyCount(newBegin, this->GetPointer(), this->GetCount());
        Free(this->GetPointer());

        this->SetPointer(newBegin);
        this->SetCount(newCount);
    }

    __host__ __device__ inline ~Array() noexcept {
        Free(this->GetPointer());
    }
}; // Array<T, D>

// Some aliases for the Array<T, D> class.
template <typename T>
using CPU_Array = Array<T, Device::CPU>;

template <typename T>
using GPU_Array = Array<T, Device::GPU>;


template <typename T>
struct Vec4 {
    T x, y, z, w;

    template <typename _H = T, typename _V = T, typename _K = T, typename _Q = T>
    __host__ __device__ inline Vec4(const _H& x = 0, const _V& y = 0, const _K& z = 0, const _Q& w = 0) noexcept {
        this->x = static_cast<T>(x); this->y = static_cast<T>(y); this->z = static_cast<T>(z); this->w = static_cast<T>(w);
    }


    __host__ __device__ inline float GetLength3D() const noexcept { return sqrt(this->x*this->x+this->y*this->y+this->z*this->z); }
    __host__ __device__ inline float GetLength4D() const noexcept { return sqrt(this->x*this->x+this->y*this->y+this->z*this->z+this->w*this->w); }

    __host__ __device__ inline void Normalize3D() noexcept { this->operator/=(this->GetLength3D()); }
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

    static __host__ __device__ inline float DotProduct3D(const Vec4<T>& a, const Vec4<T>& b) noexcept {
      return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static __host__ __device__ inline float DotProduct4D(const Vec4<T>& a, const Vec4<T>& b) noexcept {
      return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    static __host__ __device__ inline Vec4<T> Normalized3D(Vec4<T> v) noexcept {
      v.Normalize3D();
      return v;
    }

    static __host__ __device__ inline Vec4<T> Normalized4D(Vec4<T> v) noexcept {
      v.Normalize4D();
      return v;
    }
}; // Vec4<T>

template <typename T, typename _U>
__host__ __device__ inline auto operator+(const Vec4<T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x + b.x)> {
  return Vec4<decltype(a.x + b.x)>{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; 
}

template <typename T, typename _U>
__host__ __device__ inline auto operator-(const Vec4<T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x - b.x)> {
  return Vec4<decltype(a.x - b.x)>{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

template <typename T, typename _U>
__host__ __device__ inline auto operator*(const Vec4<T>& a, const Vec4<_U>& b) noexcept -> Vec4<decltype(a.x * b.x)> {
  return Vec4<decltype(a.x * b.x)>{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
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
__host__ __device__ inline auto operator*(const Vec4<T>& a, const _U& n) noexcept -> Vec4<decltype(a.x * n)> {
  return Vec4<decltype(a.x * n)>{ a.x * n, a.y * n, a.z * n, a.w * n };
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

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Vec4<T>& v) noexcept {
    stream << '(' << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ')';

    return stream;
}

typedef Vec4<std::uint8_t> Coloru8;
typedef Vec4<float>        Colorf32;
typedef Vec4<float>        Vec4f32;


__device__ inline Vec4f32 Random3DUnitVector() noexcept {
    return Vec4f32::Normalized3D(Vec4f32(2.0f * RandomFloat() - 1.0f,
                                         2.0f * RandomFloat() - 1.0f,
                                         2.0f * RandomFloat() - 1.0f,
                                         0.f));
}

template <typename T, Device D>
class Image {
private:
    std::uint16_t m_width   = 0;
    std::uint16_t m_height  = 0;
    std::uint32_t m_nPixels = 0;

    Array<T, D> m_pArray;

public:
    __host__ __device__ inline Image() = default;

    __host__ __device__ inline Image(const std::uint16_t width, const std::uint16_t height) noexcept 
        : m_width(width),
          m_height(height),
          m_nPixels(static_cast<std::uint32_t>(width) * height),
          m_pArray(Array<T, D>(this->m_nPixels))
    { }

    __host__ __device__ inline std::uint16_t GetWidth()      const noexcept { return this->m_width;   }
    __host__ __device__ inline std::uint16_t GetHeight()     const noexcept { return this->m_height;  }
    __host__ __device__ inline std::uint32_t GetPixelCount() const noexcept { return this->m_nPixels; }

    __host__ __device__ inline Pointer<T, D> GetPtr() const noexcept { return this->m_pArray; }

    __host__ __device__ inline       T& operator()(const size_t i)       noexcept { return this->m_pArray[i]; }
    __host__ __device__ inline const T& operator()(const size_t i) const noexcept { return this->m_pArray[i]; }

    __host__ __device__ inline       T& operator()(const size_t x, const size_t y)       noexcept { return this->m_pArray[y * this->m_width + this->m_height]; }
    __host__ __device__ inline const T& operator()(const size_t x, const size_t y) const noexcept { return this->m_pArray[y * this->m_width + this->m_height]; }

}; // Image

__host__ void SaveImage(const Image<Coloru8, Device::CPU>& image, const std::string& filename) noexcept {
    const std::string fullFilename = filename + ".pam";

    // Open
    FILE* fp = std::fopen(fullFilename.c_str(), "wb");

    if (fp) {
        // Header
        std::fprintf(fp, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", image.GetWidth(), image.GetHeight());

        // Write Contents
        std::fwrite(image.GetPtr(), image.GetPixelCount() * sizeof(Coloru8), 1u, fp);

        // Close
        std::fclose(fp);
    }
}

struct Camera {
    float   fov;
    Vec4f32 position;
}; // Camera

struct RenderParams {
    size_t width  = 0;
    size_t height = 0;

    Camera camera;
};

struct Ray {
    Vec4f32 origin;
    Vec4f32 direction;
}; // Ray

struct Material {
    Colorf32 diffuse;
    Colorf32 emittance;
}; // Material

struct Sphere {
    Vec4f32 center;
    float   radius;

    Material material;
}; // Sphere

struct Intersection {
    Ray      inRay;     // incoming ray
    float            t; // distance from the ray's origin to intersection point
    Vec4f32  location;  // intersection location
    Vec4f32  normal;    // normal at intersection point
    Material material;  // the material that the intersected object is made of      ::      
}; // Intersection

__device__ inline Ray GenerateCameraRay(const size_t& pixelX, const size_t& pixelY, const GPU_Ptr<RenderParams>& pRanderParams) noexcept {
    const RenderParams& renderParams = *pRanderParams;

    Ray ray;
    ray.origin    = Vec4f32(0.f, 0.f, 0.f, 0.f);
    ray.direction = Vec4f32::Normalized3D(Vec4f32(
        (2.0f *  ((pixelX + RandomFloat()) / static_cast<float>(renderParams.width))  - 1.0f) * tan(renderParams.camera.fov) * static_cast<float>(renderParams.width) / static_cast<float>(renderParams.height),
        (-2.0f * ((pixelY + RandomFloat()) / static_cast<float>(renderParams.height)) + 1.0f) * tan(renderParams.camera.fov),
        1.0f, 0.f));
    
    return ray;
}

__device__ Intersection
RayIntersects(const Ray& ray, const Sphere& sphere) noexcept {
    const Vec4f32 L   = sphere.center - ray.origin;
    const float   tca = Vec4f32::DotProduct3D(L, ray.direction);
    const float   d2  = Vec4f32::DotProduct3D(L, L) - tca * tca;
  
    if (d2 > sphere.radius)
      return Intersection{{}, FLT_MAX};
    
    const float thc = sqrt(sphere.radius - d2);
    float t0 = tca - thc;
    float t1 = tca + thc;
  
    if (t0 > t1) {
      const float tmp = t0;
      t0 = t1;
      t1 = tmp;
    }
  
    if (t0 < EPSILON) {
      t0 = t1;
  
      if (t0 < 0)
        return Intersection{{}, FLT_MAX};
    }
  
    Intersection intersection;
    intersection.inRay    = ray;
    intersection.t        = t0;
    intersection.location = ray.origin + t0 * ray.direction;
    intersection.normal   = Vec4f32::Normalized3D(intersection.location - sphere.center);
    intersection.material = sphere.material;
  
    return intersection;
}

__device__ Intersection
FindClosestIntersection(const Ray& ray,
                        const GPU_ArrayView<Sphere>& pSpheres) noexcept {
    Intersection closest;
    closest.t = FLT_MAX;

    for (size_t i = 0; i < pSpheres.GetCount(); i++) {
        const Intersection current = RayIntersects(ray, pSpheres[i]);

        if (current.t < closest.t)
            closest = current;
    }

    return closest;
}

template <size_t _N>
__device__ Colorf32 RayTrace(const Ray& ray,
                             const GPU_Ptr<RenderParams>& pParams,
                             const GPU_ArrayView<Sphere>& pSpheres) {
    auto intersection = FindClosestIntersection(ray, pSpheres);

    if constexpr (_N < MAX_REC) {
        if (intersection.t != FLT_MAX) {
            const Material& material = intersection.material;
    
            Ray newRay;
            newRay.origin    = intersection.location + EPSILON * intersection.normal;
            newRay.direction = Random3DUnitVector();
    
            return material.diffuse;
           //const Colorf32 incomingColor = RayTrace<_N + 1u>(newRay, pParams, pSpheres);
    
           //Colorf32 finalColor = material.emittance + material.diffuse * (incomingColor * 1.0f / (1.f / (2 * 3.141592f)));
           //finalColor.x = Clamp(finalColor.x, 0.f, 1.f);
           //finalColor.y = Clamp(finalColor.y, 0.f, 1.f);
           //finalColor.z = Clamp(finalColor.z, 0.f, 1.f);
           //finalColor.w = Clamp(finalColor.w, 0.f, 1.f);
    
           //return finalColor;
        }
    }

    // Sky Color
    const float ratio = (threadIdx.y + blockIdx.y * blockDim.y) / float(pParams->height);

    const auto skyLightBlue = Vec4f32(0.78f, 0.96f, 1.00f, 1.0f);
    const auto skyDarkBlue  = Vec4f32(0.01f, 0.84f, 0.93f, 1.0f);
    
    return Vec4f32(skyLightBlue * ratio + skyDarkBlue * (1 - ratio));
}

// Can't pass arguments via const& because these variables exist on the host and not on the device
__global__ void RayTracingDispatcher(const GPU_Ptr<Coloru8> pSurface,
                                        const GPU_Ptr<RenderParams> pParams,
                                        const GPU_ArrayView<Sphere> pSpheres) {
    // Calculate the thread's (X, Y) location
    const size_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

    // Bounds check
    if (pixelX >= pParams->width || pixelY >= pParams->height) return;

    // Determine the pixel's index into the image buffer
    const size_t index = pixelX + pixelY * pParams->width;

    const Ray cameraRay = GenerateCameraRay(pixelX, pixelY, pParams);

    // the current pixel's color (represented with floating point components)
    Colorf32 pixelColor = RayTrace<0>(cameraRay, pParams, pSpheres) * 255.f;

    // Save the result to the buffer
    *(pSurface + index) = Coloru8(pixelColor.x, pixelColor.y, pixelColor.z, pixelColor.w);
}

class PolarTracer {
private:
    struct {
        RenderParams m_renderParams;
    } host;

    struct {
        Image<Coloru8, Device::GPU> m_frameBuffer;
        GPU_UniquePtr<RenderParams> m_pRenderParams;
        GPU_Array<Sphere>     m_spheres;
    } device;

public:
    PolarTracer(const RenderParams& renderParams, const CPU_Array<Sphere>& spheres)
        : host{renderParams}
    {
        this->device.m_frameBuffer   = Image<Coloru8, Device::GPU>(renderParams.width, renderParams.height);
        this->device.m_pRenderParams = AllocateSingle<RenderParams, Device::GPU>();
        this->device.m_spheres       = GPU_Array<Sphere>(spheres);

        const auto src = CPU_Ptr<RenderParams>(&this->host.m_renderParams);
        CopySingle(this->device.m_pRenderParams, src);
    }

    inline void RayTraceScene(const Image<Coloru8, Device::CPU>& outSurface) {
        assert(outSurface.GetWidth() == this->host.m_renderParams.width && outSurface.GetHeight() == this->host.m_renderParams.height);

        const size_t bufferSize = outSurface.GetPixelCount() * sizeof(Coloru8);

        // Allocate 1 thread per pixel of coordinates (X,Y). Use as many blocks in the grid as needed
        // The RayTrace function will use the thread's index (both in the grid and in a block) to determine the pixel it will trace rays through
        const dim3 dimBlock = dim3(32, 32); // 32 warps of 32 threads per block (=1024 threads in total which is the hardware limit)
        const dim3 dimGrid  = dim3(std::ceil(this->host.m_renderParams.width  / static_cast<float>(dimBlock.x)),
                                   std::ceil(this->host.m_renderParams.height / static_cast<float>(dimBlock.y)));

        // trace rays through each pixel
        RayTracingDispatcher<<<dimGrid, dimBlock>>>(this->device.m_frameBuffer.GetPtr(),
                                                    this->device.m_pRenderParams,
                                                    this->device.m_spheres);
    
        // wait for the job to finish
        cudaDeviceSynchronize();

        // copy the gpu buffer to a new cpu buffer
        CopySize(outSurface.GetPtr(), this->device.m_frameBuffer.GetPtr(), bufferSize);
    }
}; // PolarTracer

#define WIDTH  (1920)
#define HEIGHT (1080)

int main(int argc, char** argv) {    
    Image<Coloru8, Device::CPU> image(WIDTH, HEIGHT);

    RenderParams renderParams;

    renderParams.width  = WIDTH;
    renderParams.height = HEIGHT;
    renderParams.camera.position = Vec4f32(0.f, 0.f, -2.f, 0.f);
    renderParams.camera.fov      = M_PI / 4.f;

    CPU_Array<Sphere> spheres(2);
    spheres[0].center = Vec4f32{0.0f, 0.0f, 2.f, 0.f};
    spheres[0].radius = 0.25f;
    spheres[0].material.diffuse   = Colorf32{1.f, 0.f, 1.f, 1.f};
    spheres[0].material.emittance = Colorf32{0.f, 0.f, 0.f, 1.f};
    
    spheres[1].center = Vec4f32{1.0f, 0.0f, 1.75f, 0.0f};
    spheres[1].radius = 0.25f;
    spheres[1].material.diffuse   = Colorf32{1.f, 1.f, 1.f, 1.f};
    spheres[1].material.emittance = Colorf32{1.f, 1.f, 1.f, 1.f};

    PolarTracer pt(renderParams, spheres);
    pt.RayTraceScene(image);

    SaveImage(image, "frame");
}