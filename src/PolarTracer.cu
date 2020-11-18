#include <array>
#include <thread>
#include <cfloat>
#include <chrono>
#include <future>
#include <limits>
#include <ostream>
#include <cassert>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <optional>
#include <iostream>

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <Windows.h>
#include <windowsx.h>
#include <d3dcompiler.h>

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "user32.lib") 
#pragma comment(lib, "kernel32.lib")
#pragma comment(lib, "D3DCompiler.lib")

// +--------------------+
// | Constants / Common |
// +--------------------+

__constant__ const float  EPSILON = 0.0001f;
__constant__ const size_t MAX_REC = 3u;
__constant__ const size_t SPP     = 5u;

// +--------+
// | Window |
// +--------+

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept;

class Window {
private:
    HWND m_handle = NULL;

    size_t m_width, m_height;

public:
    Window() = default;

    Window(const char* name, const size_t width, const size_t height) noexcept
        : m_width(width), m_height(height)
    {
        const HINSTANCE hInstance = GetModuleHandleA(NULL);

        WNDCLASSA wc = { };
        wc.lpfnWndProc = WindowProc;
        wc.hInstance = hInstance;
        wc.lpszClassName = name;
        RegisterClassA(&wc);

        RECT rect{ 0, 0, width, height };
        if (!AdjustWindowRect(&rect, NULL, false))
            return;

        this->m_handle = CreateWindowExA(
            0, name,
            name, WS_OVERLAPPEDWINDOW ^ WS_THICKFRAME, // unresizable
            CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top,
            NULL, NULL, hInstance, NULL
        );

        if (this->m_handle == NULL)
            return;

        ShowWindow(this->m_handle, SW_SHOW);

#ifdef _WIN64
        SetWindowLongPtrA(this->m_handle, GWLP_USERDATA, (LONG_PTR)this);
#else
        SetWindowLongA(this->m_handle, GWLP_USERDATA, (LONG)this);
#endif
    }

    inline void Update() noexcept {
        MSG msg = { };
        while (PeekMessageA(&msg, this->m_handle, 0, 0, PM_REMOVE) > 0) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
    }

    inline size_t GetClientWidth()  const noexcept { return this->m_width; }
    inline size_t GetClientHeight() const noexcept { return this->m_height; }

    inline HWND GetHandle() const noexcept { return this->m_handle; }
    inline bool IsRunning() const noexcept { return this->m_handle != NULL; }

    inline void Close() noexcept {
        DestroyWindow(this->m_handle);

        this->m_handle = NULL;
    }

    inline ~Window() noexcept { this->Close(); }
}; // class Window

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) noexcept {
#ifdef _WIN64
    Window* pWindow = reinterpret_cast<Window*>(GetWindowLongPtrA(hwnd, GWLP_USERDATA));
#else
    Window* pWindow = reinterpret_cast<Window*>(GetWindowLongA(hwnd, GWLP_USERDATA));
#endif

    if (pWindow) {
        switch (msg) {
        case WM_CLOSE:
            pWindow->Close();
            return 0;
        }
    }

    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

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
}; // namespace Utility

// +--------+
// | Memory |
// +--------+

namespace Memory {
    template <typename T, Device D> class Pointer;
    template <typename T, Device D> class UniquePointer;
    template <typename T, Device D> class ArrayView;
    template <typename T, Device D> class Array;

    template <typename T> using CPU_Ptr       = Pointer<T,       Device::CPU>;
    template <typename T> using GPU_Ptr       = Pointer<T,       Device::GPU>;
    template <typename T> using CPU_UniquePtr = UniquePointer<T, Device::CPU>;
    template <typename T> using GPU_UniquePtr = UniquePointer<T, Device::GPU>;
    template <typename T> using CPU_ArrayView = ArrayView<T,     Device::CPU>;
    template <typename T> using GPU_ArrayView = ArrayView<T,     Device::GPU>;
    template <typename T> using CPU_Array     = Array<T,         Device::CPU>;
    template <typename T> using GPU_Array     = Array<T,         Device::GPU>;

    template <typename T, Device D> inline Pointer<T, D> AllocateSize   (const size_t size)  noexcept;
    template <typename T, Device D> inline Pointer<T, D> AllocateCount  (const size_t count) noexcept;
    template <typename T, Device D> inline Pointer<T, D> AllocateSingle ()                   noexcept;

    template <typename T, Device D> inline void Free(const Pointer<T, D>& p) noexcept;

    template <template<typename, Device> typename _PTR_DST, typename T_DST, Device D_DST,
              template<typename, Device> typename _PTR_SRC, typename T_SRC, Device D_SRC>
    inline void CopySize(const _PTR_DST<T_DST, D_DST>& dst, const _PTR_SRC<T_SRC, D_SRC>& src, const size_t size) noexcept;

    template <template<typename, Device> typename _PTR_DST, typename T_DST, Device D_DST,
              template<typename, Device> typename _PTR_SRC, typename T_SRC, Device D_SRC>
    inline void CopyCount(const _PTR_DST<T_DST, D_DST>& dst, const _PTR_SRC<T_SRC, D_SRC>& src, const size_t count) noexcept;

    template <template<typename, Device> typename _PTR_DST, typename T_DST, Device D_DST,
              template<typename, Device> typename _PTR_SRC, typename T_SRC, Device D_SRC>
    inline void CopySingle(const _PTR_DST<T_DST, D_DST>& dst, const _PTR_SRC<T_SRC, D_SRC>& src) noexcept;
}; // Memory Memory

namespace Memory {
    template <typename T, Device D>
    class Pointer {
    private:
        T* m_raw = nullptr;

    public:
        Pointer() = default;

        __host__ __device__ inline Pointer(T* const p)             noexcept { this->SetPointer(p); }
        __host__ __device__ inline Pointer(const Pointer<T, D>& o) noexcept { this->SetPointer(o); }

        __host__ __device__ inline void SetPointer(T* const p)             noexcept { this->m_raw = p;           }
        __host__ __device__ inline void SetPointer(const Pointer<T, D>& o) noexcept { this->SetPointer(o.m_raw); }

        __host__ __device__ inline T*& GetPointer()       noexcept { return this->m_raw; }
        __host__ __device__ inline T*  GetPointer() const noexcept { return this->m_raw; }

        template <typename U>
        __host__ __device__ inline Pointer<U, D> AsPointerTo() const noexcept { return Pointer<U, D>(reinterpret_cast<U*>(this->m_raw)); }

        __host__ __device__ inline void operator=(T* const p)             noexcept { this->SetPointer(p); }
        __host__ __device__ inline void operator=(const Pointer<T, D>& o) noexcept { this->SetPointer(o); }

        __host__ __device__ inline operator T*& ()       noexcept { return this->m_raw; }
        __host__ __device__ inline operator T*  () const noexcept { return this->m_raw; }

        __host__ __device__ inline std::conditional_t<std::is_same_v<T, void>, int, T>& operator[](const size_t i) noexcept {
            static_assert(!std::is_same_v<T, void>, "Can't Index A Pointer To A Void");
            return *(this->m_ptr + i);
        }

        __host__ __device__ inline const std::conditional_t<std::is_same_v<T, void>, int, T>& operator[](const size_t i) const noexcept {
            static_assert(!std::is_same_v<T, void>, "Can't Index A Pointer To A Void");
            return *(this->m_ptr + i);
        }

        __host__ __device__ inline T* operator->() const noexcept { return this->m_raw; }
    }; // class Pointer<T, D>

    template <typename T, Device D>
    class UniquePointer {
    private:
        Pointer<T, D> m_ptr;
    
    public:
        inline UniquePointer() noexcept {
            this->Free();
            this->m_ptr = nullptr;
        }
    
        template <typename... _ARGS>
        inline UniquePointer(const _ARGS&... args) noexcept {
            this->Free();
            this->m_ptr = new (AllocateSingle<T, D>()) T(std::forward<_ARGS>(args)...);
        }
    
        inline UniquePointer(const Pointer<T, D>& o)  noexcept {
            this->Free();
            this->m_ptr = o;
        }
    
        inline UniquePointer(UniquePointer<T, D>&& o) noexcept {
            this->Free();
            this->m_ptr = o.m_ptr;
            o.m_ptr = nullptr;
        }
    
        inline void Free() const noexcept {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                // Since we use placement new, we have to call T's destructor ourselves
                this->m_ptr->~T();
            }
    
            // and then free the memory
            ::Memory::Free(this->m_ptr);
        }
    
        inline ~UniquePointer() noexcept { this->Free(); }
    
        inline UniquePointer<T, D>& operator=(const Pointer<T, D>& o)  noexcept {
            this->Free();
            this->m_ptr = o;
            return *this;
        }
    
        inline UniquePointer<T, D>& operator=(UniquePointer<T, D>&& o) noexcept {
            this->Free();
            this->m_ptr = o.m_ptr;
            o.m_ptr = nullptr;
            return *this;
        }
    
        __host__ __device__ inline const Pointer<T, D>& GetPointer() const noexcept { return this->m_ptr; }
    
        template <typename U>
        __host__ __device__ inline Pointer<U, D> AsPointerTo() const noexcept { return this->m_ptr.AsPointerTo<U>(); }

        __host__ __device__ inline operator const Pointer<T, D>& () const noexcept { return this->m_ptr; }
    
        __host__ __device__ inline operator T* () const noexcept { return this->m_ptr; }
    
        __host__ __device__ inline T* operator->() const noexcept { return this->m_ptr; }
    
        __host__ __device__ inline       T& operator[](const size_t i)       noexcept { return *(this->m_ptr + i); }
        __host__ __device__ inline const T& operator[](const size_t i) const noexcept { return *(this->m_ptr + i); }
    
    
        UniquePointer(const UniquePointer<T, D>& o) = delete;
        UniquePointer<T, D>& operator=(const UniquePointer<T, D>& o) = delete;
    }; // class UniquePointer<T, D>
    
    template <typename T, Device D>
    class ArrayView {
    public:
        using VALUE_TYPE = T;
        constexpr static Device DEVICE = D;

    protected:
        Pointer<T, D> m_pBegin;

        size_t m_count    = 0;
        size_t m_capacity = 0;

    protected:
        __host__ __device__ inline operator T*& () noexcept { return this->m_pBegin; }

        __host__ __device__ inline ArrayView(const Pointer<T, D>& pBegin, const size_t count, const size_t capacity) noexcept
            : m_pBegin(pBegin), m_count(count), m_capacity(capacity)
        {  }

    public:
        ArrayView() = default;

        inline ArrayView(const Pointer<T, D>& pBegin, const size_t count) noexcept
            : m_pBegin(pBegin), m_capacity(count)
        {  }

        __host__ __device__ inline Pointer<T, D> begin() const noexcept { return this->m_pBegin;                 }
        __host__ __device__ inline Pointer<T, D> end()   const noexcept { return this->m_pBegin + this->m_count; }

        __host__ __device__ inline const Pointer<T, D>& GetPointer()  const noexcept { return this->m_pBegin;   }
        __host__ __device__ inline const size_t&        GetCount()    const noexcept { return this->m_count;    }
        __host__ __device__ inline const size_t&        GetCapacity() const noexcept { return this->m_capacity; }

        __host__ __device__ inline operator const Pointer<T, D>& () const noexcept { return this->m_pBegin; }

        __host__ __device__ inline operator T* () const noexcept { return this->m_pBegin; }

        __host__ __device__ inline       T& operator[](const size_t i)       noexcept { return *(this->m_pBegin + i); }
        __host__ __device__ inline const T& operator[](const size_t i) const noexcept { return *(this->m_pBegin + i); }

        ~ArrayView() = default;
    }; // class ArrayView<T, D>

    template <typename T, Device D>
    class Array : public ArrayView<T, D> {
    private:
        inline void Free() noexcept { ::Memory::Free(this->m_pBegin); }

    public:
        inline Array() noexcept = default;

        inline Array(const size_t count) noexcept
            : ArrayView<T, D>(AllocateCount<T, D>(count), count, count)
        {  }

        template <Device D_O>
        inline Array(const Array<T, D_O>& o) noexcept
            : Array(o.GetCount())
        {
            CopyCount(*this, o, this->GetCapacity());
        }

        template <typename T2, Device D_O>
        inline Array(const Array<T2, D_O>& o) noexcept
            : Array(o.GetCount())
        {
            for (size_t i = 0u; i < this->GetCount(); ++i)
                (*this)[i] = o[i];
        }

        inline Array(Array<T, D>&& o) noexcept {
            this->m_capacity = o.m_capacity;
            this->m_count    = o.m_count;
            this->m_pBegin   = o.m_pBegin;
            o.m_pBegin       = nullptr;
        }

        inline Array<T, D>& operator=(Array<T, D>&& o) noexcept {
            this->Free();

            this->m_capacity = o.m_capacity;
            this->m_count    = o.m_count;
            this->m_pBegin   = o.m_pBegin;
            o.m_pBegin       = nullptr;

            return *this;
        }

        inline void Reserve(const size_t count) noexcept {
            const auto newCapacity = this->m_capacity + count;
            const auto newBegin    = AllocateCount<T, D>(newCapacity);

            CopyCount(newBegin, this->GetPointer(), this->GetCapacity());
            this->Free();

            this->m_pBegin   = newBegin;
            this->m_capacity = newCapacity;
        }

        template <typename U>
        inline void Append(U&& e) noexcept {
            if (this->m_count >= this->m_capacity)
                this->Reserve(1);

            (*this)[this->m_count++] = std::forward<U>(e);
        }

        template <typename... Args>
        inline void AppendArgs(const Args&&... args) noexcept {
            this->Append(T(std::forward<Args>(args)...));
        }

        inline ~Array() noexcept { this->Free(); }
    }; // class Array<T, D>

    template <typename T, Device D>
    inline Pointer<T, D> AllocateSize(const size_t size) noexcept {
        if constexpr (D == Device::CPU) {
            return CPU_Ptr<T>(reinterpret_cast<T*>(std::malloc(size)));
        } else {
            T* p;
            cudaMalloc(reinterpret_cast<void**>(&p), size);
            return GPU_Ptr<T>(p);
        }

        return Pointer<T, D>{nullptr}; // done to suppress a warning
    }

    template <typename T, Device D>
    inline Pointer<T, D> AllocateCount(const size_t count) noexcept {
        return AllocateSize<T, D>(count * sizeof(T));
    }

    template <typename T, Device D>
    inline Pointer<T, D> AllocateSingle() noexcept {
        return AllocateSize<T, D>(sizeof(T));
    }

    template <typename T, Device D>
    inline void Free(const Pointer<T, D>& p) noexcept {
        if constexpr (D == Device::CPU) {
            std::free(p);
        } else {
            cudaFree(p.template AsPointerTo<void>());
        }
    }

    template <template<typename, Device> typename _PTR_DST, typename T_DST, Device D_DST,
              template<typename, Device> typename _PTR_SRC, typename T_SRC, Device D_SRC>
    inline void CopySize(const _PTR_DST<T_DST, D_DST>& dst, const _PTR_SRC<T_SRC, D_SRC>& src, const size_t size) noexcept
    {
        static_assert(std::is_same_v<T_DST, T_SRC>, "Incompatible Source And Destination Raw Pointer Types");

        cudaMemcpyKind memcpyKind;

        if constexpr (D_SRC == Device::CPU && D_DST == Device::CPU) {
            memcpyKind = cudaMemcpyKind::cudaMemcpyHostToHost;
            cudaMemcpy(dst, src, size, memcpyKind);
        } else if constexpr (D_SRC == Device::GPU && D_DST == Device::GPU) {
            memcpyKind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
            cudaMemcpy(dst, src, size, memcpyKind);
        } else if constexpr (D_SRC == Device::CPU && D_DST == Device::GPU) {
            memcpyKind = cudaMemcpyKind::cudaMemcpyHostToDevice;
            cudaMemcpy(dst, src, size, memcpyKind);
        } else if constexpr (D_SRC == Device::GPU && D_DST == Device::CPU) {
            memcpyKind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
            cudaMemcpy(dst, src, size, memcpyKind);
        } else { static_assert(1 == 1, "Incompatible Destination and Source Arguments"); }
    }

    template <template<typename, Device> typename _PTR_DST, typename T_DST, Device D_DST,
              template<typename, Device> typename _PTR_SRC, typename T_SRC, Device D_SRC>
    inline void CopyCount(const _PTR_DST<T_DST, D_DST>& dst, const _PTR_SRC<T_SRC, D_SRC>& src, const size_t count) noexcept
    {
        static_assert(std::is_same_v<T_DST, T_SRC>, "Incompatible Source And Destination Raw Pointer Types");

        CopySize(dst, src, count * sizeof(T_DST));
    }

    template <template<typename, Device> typename _PTR_DST, typename T_DST, Device D_DST,
              template<typename, Device> typename _PTR_SRC, typename T_SRC, Device D_SRC>
    inline void CopySingle(const _PTR_DST<T_DST, D_DST>& dst, const _PTR_SRC<T_SRC, D_SRC>& src) noexcept
    {
        static_assert(std::is_same_v<T_DST, T_SRC>, "Incompatible Source And Destination Raw Pointer Types");

        CopySize(dst, src, sizeof(T_DST));
    }

}; // namespace Memory

__device__ inline float RandomFloat(curandState_t* randState) noexcept {
    return curand_uniform(randState);
}

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
    
        __host__ __device__ inline float GetLength3D() const noexcept { return sqrt(this->x * this->x + this->y * this->y + this->z * this->z); }
        __host__ __device__ inline float GetLength4D() const noexcept { return sqrt(this->x * this->x + this->y * this->y + this->z * this->z + this->w * this->w); }
    
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
        return Normalized3D(Vec4<T>(2.0f * RandomFloat(pRandSate) - 1.0f,
                                    2.0f * RandomFloat(pRandSate) - 1.0f,
                                    2.0f * RandomFloat(pRandSate) - 1.0f,
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

// +-------+
// | Image |
// +-------+

template <class Container>
class ImageBase {
private:
    std::uint16_t m_width   = 0;
    std::uint16_t m_height  = 0;
    std::uint32_t m_nPixels = 0;

    Container m_container;

public:
    ImageBase() = default;

    template <class Container2>
    inline ImageBase(const ImageBase<Container2>& o) noexcept {
        this->m_width   = o.GetWidth();
        this->m_height  = o.GetHeight();
        this->m_nPixels = o.GetPixelCount();
        this->m_container = o.GetContainer();
    }

    inline ImageBase(const std::uint16_t width, const std::uint16_t height) noexcept {
        this->m_width     = width;
        this->m_height    = height;
        this->m_nPixels   = static_cast<std::uint32_t>(width) * height;
        this->m_container = Memory::Array<typename Container::VALUE_TYPE, Container::DEVICE>(this->m_nPixels);
    }

    __host__ __device__ inline std::uint16_t GetWidth()      const noexcept { return this->m_width;   }
    __host__ __device__ inline std::uint16_t GetHeight()     const noexcept { return this->m_height;  }
    __host__ __device__ inline std::uint32_t GetPixelCount() const noexcept { return this->m_nPixels; }

    __host__ __device__ inline       Container& GetContainer()       noexcept { return this->m_container; }
    __host__ __device__ inline const Container& GetContainer() const noexcept { return this->m_container; }
    __host__ __device__ inline Memory::Pointer<typename Container::VALUE_TYPE, Container::DEVICE> GetPtr() const noexcept { return this->m_container; }

    __host__ __device__ inline       typename Container::VALUE_TYPE& operator()(const size_t i)       noexcept { return this->m_container[i]; }
    __host__ __device__ inline const typename Container::VALUE_TYPE& operator()(const size_t i) const noexcept { return this->m_container[i]; }

    __host__ __device__ inline       typename Container::VALUE_TYPE& operator()(const size_t x, const size_t y)       noexcept { return this->m_container[y * this->m_width + this->m_height]; }
    __host__ __device__ inline const typename Container::VALUE_TYPE& operator()(const size_t x, const size_t y) const noexcept { return this->m_container[y * this->m_width + this->m_height]; }
}; // Image

template <typename T, Device D> using Image     = ImageBase<Memory::Array<T, D>>;
template <typename T, Device D> using ImageView = ImageBase<Memory::ArrayView<T, D>>; // ImageView

template <typename T> using CPU_Image = Image<T, Device::CPU>;
template <typename T> using GPU_Image = Image<T, Device::GPU>;

template <typename T> using CPU_ImageView = ImageView<T, Device::CPU>;
template <typename T> using GPU_ImageView = ImageView<T, Device::GPU>;

template <typename T, Device D> using Texture = Image<T, D>;
template <typename T> using CPU_Texture = CPU_Image<T>;
template <typename T> using GPU_Texture = GPU_Image<T>;

CPU_Image<Math::Coloru8> ReadImage(const std::string& filename) noexcept {
    std::ifstream file(filename.c_str(), std::ifstream::binary);

    std::cout << "created\n";

	if (file.is_open()) {
        std::cout << "opened\n";
		// Read Header
		std::string magicNumber;
		float maxColorValue;

        std::uint16_t width, height;
		file >> magicNumber >> width >> height >> maxColorValue;

		file.ignore(1); // Skip the last whitespace
        
        CPU_Image<Math::Coloru8> image(width, height);

		// Parse Image Data
        if (magicNumber == "P6") {
            std::cout << "read magic\n";
            const size_t bufferSize = image.GetPixelCount() * sizeof(Math::Coloru8);

            const Memory::CPU_UniquePtr<std::uint8_t> rawImageData = Memory::AllocateSize<std::uint8_t, Device::CPU>(bufferSize);
            file.read(rawImageData.AsPointerTo<char>(), bufferSize);

            size_t j = 0;
            for (size_t i = 0; i < image.GetPixelCount(); ++i)
                image(i) = Math::Coloru8(rawImageData[j++], rawImageData[j++], rawImageData[j++], 255u);

            std::cout << "done\n";
        } else {
			std::cout << magicNumber << " is not supported\n";
        }

        std::cout << "closing\n";

        file.close();

        std::cout << "closed\n";

        return image;
	} else {
		perror("Error while reading image file");
    }

    return CPU_Image<Math::Coloru8>(0, 0);
}

// +-------------+
// | PolarTracer |
// +-------------+

struct Camera {
    float         fov;
    Math::Vec4f32 position;
}; // Camera

struct RenderParams {
    size_t width  = 0;
    size_t height = 0;

    Camera camera;
};

struct Intersection;

struct Ray {
    Math::Vec4f32 origin;
    Math::Vec4f32 direction;

    template <typename _T_OBJ>
    __device__ Intersection Intersects(const _T_OBJ& obj) const noexcept;
}; // Ray

struct Material {
    Math::Colorf32 diffuse   = { 0.f, 0.f, 0.f, 1.f };
    Math::Colorf32 emittance = { 0.f, 0.f, 0.f, 1.f };

    float reflectance  = 0.f; // the sum of these values should be less than or equal to 1
    float transparency = 0.f; // the sum of these values should be less than or equal to 1

    float roughness = 0.f;

    float index_of_refraction = 1.f;
}; // Material

struct SphereGeometry {
    Math::Vec4f32 center;
    float         radius;
}; // SphereGeometry

struct Sphere : SphereGeometry {
    Material material;
}; // Sphere

struct PlaneGeometry {
    Math::Vec4f32 position; // Any Point On The Plane
    Math::Vec4f32 normal;   // Normal To The Surface
}; // PlaneGeometry

struct Plane : PlaneGeometry {
    Material material;
}; // Plane

struct TriangleGeometry {
    Math::Vec4f32 v0; // Position of the 1st vertex
    Math::Vec4f32 v1; // Position of the 2nd vertex
    Math::Vec4f32 v2; // Position of the 3rd vertex
}; // TriangleGeometry

struct Triangle : TriangleGeometry {
    Material material;
}; // Triangle

struct BoxGeometry {
    Math::Vec4f32 center;
    float         sideLength;
}; // BoxGeometry

struct Box : BoxGeometry {
    Material material;
}; // Box

struct Intersection {
    Ray           inRay;     // incoming ray
    float         t;         // distance from the ray's origin to intersection point
    Math::Vec4f32 location;  // intersection location
    Math::Vec4f32 normal;    // normal at intersection point
    Material material;       // the material that the intersected object is made of

    __device__ __host__ inline bool operator<(const Intersection& o) const noexcept {
    	return this->t < o.t;
    }

    __device__ __host__ inline bool operator>(const Intersection& o) const noexcept {
    	return this->t > o.t;
    }

    static __device__ __host__ inline Intersection MakeNullIntersection(const Ray& ray) noexcept {
        return Intersection{ray, FLT_MAX};
    }
}; // Intersection

template <>
__device__ Intersection Ray::Intersects(const Sphere& sphere) const noexcept {
    const float radius2 = sphere.radius * sphere.radius;

    const auto L   = sphere.center - this->origin;
    const auto tca = Math::DotProduct3D(L, this->direction);
    const auto d2  = Math::DotProduct3D(L, L) - tca * tca;

    if (d2 > radius2)
        return Intersection::MakeNullIntersection(*this);

    const float thc = sqrt(radius2 - d2);
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
            return Intersection::MakeNullIntersection(*this);
    }

    Intersection intersection;
    intersection.inRay    = *this;
    intersection.t        = t0;
    intersection.location = this->origin + t0 * this->direction;
    intersection.normal   = Math::Normalized3D(intersection.location - sphere.center);
    intersection.material = sphere.material;

    return intersection;
}

template <>
__device__ Intersection Ray::Intersects(const Plane& plane) const noexcept {
    const float denom = Math::DotProduct3D(plane.normal, this->direction);
    if (abs(denom) >= EPSILON) {
        const auto  v = plane.position - this->origin;
        const float t = Math::DotProduct3D(v, plane.normal) / denom;

        if (t >= 0) {
            Intersection intersection;
            intersection.inRay    = *this;
            intersection.t        = t;
            intersection.location = this->origin + t * this->direction;
            intersection.normal   = plane.normal;
            intersection.material = plane.material;

            return intersection;
        }
    }

    return Intersection::MakeNullIntersection(*this);
}

//https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle
template <>
__device__ Intersection Ray::Intersects(const Triangle& triangle) const noexcept {
    float u = 0.f, v = 0.f;

    const auto v0v1 = triangle.v1 - triangle.v0; 
    const auto v0v2 = triangle.v2 - triangle.v0; 
    const auto pvec = Math::CrossProduct3D(v0v2,  this->direction);

    const auto det = Math::DotProduct3D(v0v1, pvec); 

    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < EPSILON) return Intersection::MakeNullIntersection(*this);

    const float invDet = 1.f / det; 
 
    const auto tvec = this->origin - triangle.v0; 
    u = Math::DotProduct3D(tvec, pvec) * invDet; 
    if (u < 0 || u > 1) return Intersection::MakeNullIntersection(*this); 
 
    const auto qvec = Math::CrossProduct3D(tvec, v0v1); 
    v = Math::DotProduct3D(this->direction, qvec) * invDet; 
    if (v < 0 || u + v > 1) return Intersection::MakeNullIntersection(*this); 
 
    Intersection intersection;
    intersection.t        = Math::DotProduct3D(v0v2, qvec) * invDet;

    if (intersection.t <= 0) return Intersection::MakeNullIntersection(*this); 

    intersection.inRay    = *this;
    intersection.location = this->origin + intersection.t * this->direction;
    intersection.normal   = Math::Normalized3D(Math::CrossProduct3D(v0v1, v0v2)); //TODO:: Actual Normal
    intersection.material = triangle.material;

    return intersection;
}

__device__ inline Ray GenerateCameraRay(const size_t& pixelX, const size_t& pixelY, const Memory::GPU_Ptr<RenderParams>& pRanderParams, curandState_t* const randState) noexcept {
    const RenderParams& renderParams = *pRanderParams;

    Ray ray;
    ray.origin = renderParams.camera.position;
    ray.direction = Math::Normalized3D(Math::Vec4f32(
        (2.0f  * ((pixelX + RandomFloat(randState)) / static_cast<float>(renderParams.width))  - 1.0f) * tan(renderParams.camera.fov) * static_cast<float>(renderParams.width) / static_cast<float>(renderParams.height),
        (-2.0f * ((pixelY + RandomFloat(randState)) / static_cast<float>(renderParams.height)) + 1.0f) * tan(renderParams.camera.fov),
        1.0f,
        0.f));

    return ray;
}

template <typename T_OBJ>
__device__ inline void FindClosestIntersection(const Ray& ray,
                                               Intersection& closest, // in/out
                                               const Memory::GPU_ArrayView<T_OBJ>& objArrayView) noexcept {
    for (const T_OBJ& obj : objArrayView) {
        const Intersection current = ray.Intersects(obj);

        if (current < closest)
            closest = current;
    }
}

template <template<typename, Device> typename Container, Device D>
struct Primitives {
    Container<Sphere, D>   spheres;
    Container<Plane, D>    planes;
    Container<Triangle, D> triangles;

    Primitives() = default;

    template <Device D_2>
    inline Primitives(const Primitives<Container, D_2>& o) noexcept {
        this->spheres   = o.spheres;
        this->planes    = o.planes;
        this->triangles = o.triangles;
    }

    template <template<typename, Device> typename C_2, Device D_2>
    inline Primitives(const Primitives<C_2, D_2>& o) noexcept {
        this->spheres   = o.spheres;
        this->planes    = o.planes;
        this->triangles = o.triangles;
    }
}; // Primitives

__device__ Intersection
FindClosestIntersection(const Ray& ray, const Primitives<Memory::ArrayView, Device::GPU>& primitives) noexcept {
    Intersection closest;
    closest.t = FLT_MAX;
    
    FindClosestIntersection(ray, closest, primitives.spheres);
    FindClosestIntersection(ray, closest, primitives.planes);
    FindClosestIntersection(ray, closest, primitives.triangles);

    return closest;
}

template <size_t _N>
__device__ Math::Colorf32 RayTrace(const Ray& ray,
                                   const Primitives<Memory::ArrayView, Device::GPU>& primitives,
                                   curandState_t* const randState) {
    const auto intersection = FindClosestIntersection(ray, primitives);

    if constexpr (_N < MAX_REC) {
        if (intersection.t != FLT_MAX) {
            const Material& material = intersection.material;
            
            Ray newRay;
            newRay.origin = intersection.location + EPSILON * intersection.normal;
            
            const float rngd = RandomFloat(randState);

            if (material.reflectance > rngd) {
                // Compute Reflexion
                newRay.direction = material.roughness * Math::Random3DUnitVector<float>(randState) + (1 - material.roughness) * Math::Reflected3D(ray.direction, intersection.normal);
            } else if (material.transparency + material.reflectance > rngd) {
                // Compute Transparency
                const bool outside = Math::DotProduct3D(ray.direction, intersection.normal) < 0;

                newRay.direction = Math::Normalized3D(Math::Refracted(ray.direction, intersection.normal, material.index_of_refraction));
                newRay.origin    = intersection.location + (outside ? -1 : 1) * EPSILON * intersection.normal;
            } else {
                // Compute Diffuse
                newRay.direction = Math::Random3DUnitVector<float>(randState);
            }
            
            const auto materialComp = RayTrace<_N + 1u>(newRay, primitives, randState);
            const auto finalColor   = material.emittance + material.diffuse * materialComp;

            return finalColor;
        }
    }
    
    // Black
    return blackf32;
}

// Can't pass arguments via const& because these variables exist on the host and not on the device
__global__ void RayTracingDispatcher(const Memory::GPU_Ptr<Math::Coloru8> pSurface,
                                     const Memory::GPU_Ptr<RenderParams> pParams,
                                     const Primitives<Memory::ArrayView, Device::GPU> primitives) {
    // Calculate the thread's (X, Y) location
    const std::uint64_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    const std::uint64_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

    // Bounds check
    if (pixelX >= pParams->width || pixelY >= pParams->height) return;

    curandState_t randState;
    curand_init(pixelX, pixelY, 0, &randState);

    const Ray cameraRay = GenerateCameraRay(pixelX, pixelY, pParams, &randState);

    // the current pixel's color (represented with floating point components)
    Math::Colorf32 pixelColor{};
    for (size_t i = 0; i < SPP; i++)
        pixelColor += RayTrace<0>(cameraRay, primitives, &randState);
    
    pixelColor *= 255.f / static_cast<float>(SPP);
    pixelColor.Clamp(0.f, 255.f);

    // Determine the pixel's index into the image buffer
    const size_t index = pixelX + pixelY * pParams->width;

    // Save the result to the buffer
    *(pSurface + index) = Math::Coloru8(pixelColor.x, pixelColor.y, pixelColor.z, pixelColor.w);
}

#define FRAME_BUFFER_COUNT (2)

template <typename T>
using ElementPerFrameBuffer = std::array<T, FRAME_BUFFER_COUNT>;

class PolarTracer {
private:
    IDXGIFactory4*        m_dxgiFactory               = nullptr;
    ID3D12Device2*        m_device                    = nullptr;
    IDXGISwapChain3*      m_swapChain                 = nullptr;
    ID3D12CommandQueue*   m_commandQueue              = nullptr;
    ID3D12DescriptorHeap* m_frameBufferDescriptorHeap = nullptr;

    ElementPerFrameBuffer<ID3D12Fence*>               m_fences;
    ElementPerFrameBuffer<UINT>                       m_fenceValues = { 0 };
    ElementPerFrameBuffer<ID3D12Resource*>            m_frameBuffers;
    ElementPerFrameBuffer<ID3D12CommandAllocator*>    m_commandAllocators;
    ElementPerFrameBuffer<ID3D12GraphicsCommandList*> m_commandLists;

    HANDLE m_fenceEvent;

    ElementPerFrameBuffer<ID3D12Resource*> m_textures;

private:
    struct {
        RenderParams m_renderParams;
    } host;

    struct {
        GPU_Image<Math::Coloru8> m_frameBuffer;
        Memory::GPU_UniquePtr<RenderParams> m_pRenderParams;

        Primitives<Memory::Array, Device::GPU> m_primitives;
        Memory::GPU_Array<GPU_Image<Math::Coloru8>> m_textures;
    } device;

    Window window;

public:
    PolarTracer(const RenderParams& renderParams, const Primitives<Memory::Array, Device::CPU>& primitives = {}, const Memory::CPU_Array<CPU_Image<Math::Coloru8>>& textures = {})
        : host{ renderParams }, window("PolarTracer", 1920, 1080)
    {
        // Save arguments
        this->device.m_pRenderParams = Memory::AllocateSingle<RenderParams, Device::GPU>();
        this->device.m_primitives    = primitives;
        this->device.m_textures      = textures;

        const auto src = Memory::CPU_Ptr<RenderParams>(&this->host.m_renderParams);
        Memory::CopySingle(this->device.m_pRenderParams, src);

        // 
        this->device.m_frameBuffer   = Image<Math::Coloru8, Device::GPU>(renderParams.width, renderParams.height);

        // Initialize DirectX 12
        // Create DXGIFactory
        if (CreateDXGIFactory2(0, IID_PPV_ARGS(&this->m_dxgiFactory)) != S_OK)
            std::cout << "[D3D12] Could Not Create DXGIFactory2\n";

        // Create Device
        IDXGIAdapter1* adapter;
        for (UINT adapterIndex = 0; this->m_dxgiFactory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND; ++adapterIndex) {
            DXGI_ADAPTER_DESC1 desc;
            if (adapter->GetDesc1(&desc) != S_OK)
                continue;

            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                continue;

            if (D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr) == S_OK)
                break;
        }

        if (D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device)) != S_OK)
            std::cout << "[DIRECTX 12] Failed To D3D12CreateDevice\n";

        // Commad Queue
        D3D12_COMMAND_QUEUE_DESC cmdQueueDesc = {};
        cmdQueueDesc.Type = D3D12_COMMAND_LIST_TYPE::D3D12_COMMAND_LIST_TYPE_DIRECT;
        cmdQueueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        cmdQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        cmdQueueDesc.NodeMask = 0;

        if (this->m_device->CreateCommandQueue(&cmdQueueDesc, IID_PPV_ARGS(&this->m_commandQueue)) != S_OK)
            std::cout << "[D3D12] Could Not Create Command Queue";

        // Swapchain
        DXGI_MODE_DESC backBufferDesc = {};
        backBufferDesc.Width = window.GetClientWidth();
        backBufferDesc.Height = window.GetClientHeight();
        backBufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

        DXGI_SAMPLE_DESC sampleDesc = {};
        sampleDesc.Count = 1;

        DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
        swapChainDesc.BufferCount = FRAME_BUFFER_COUNT;
        swapChainDesc.BufferDesc = backBufferDesc;
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        swapChainDesc.OutputWindow = window.GetHandle();
        swapChainDesc.SampleDesc = sampleDesc;
        swapChainDesc.Windowed = true;
        swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING; // 0

        if (this->m_dxgiFactory->CreateSwapChain(this->m_commandQueue, &swapChainDesc, (IDXGISwapChain**)&this->m_swapChain) != S_OK)
            std::cout << "[DIRECTX 12] Failed To Create Swap Chain";

        // Create Frame Buffer Descriptor Heap

        D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
        descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE::D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        descriptorHeapDesc.NumDescriptors = FRAME_BUFFER_COUNT;
        descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAGS::D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

        if (this->m_device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&this->m_frameBufferDescriptorHeap)) != S_OK)
            std::cout << "[D3D12] Could Not Create Descriptor Heap";

        // Get Frame Buffers & Create Render Target Views
        D3D12_CPU_DESCRIPTOR_HANDLE currentLocation = this->m_frameBufferDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
        const UINT incrementSize = this->m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        for (size_t frameIndex = 0u; frameIndex < FRAME_BUFFER_COUNT; ++frameIndex) {
            if (this->m_swapChain->GetBuffer(frameIndex, IID_PPV_ARGS(&this->m_frameBuffers[frameIndex])) != S_OK)
                std::cout << "[DIRECTX 12] Failed To Get Back Buffer";

            this->m_device->CreateRenderTargetView(this->m_frameBuffers[frameIndex], nullptr, currentLocation);
            currentLocation.ptr += incrementSize;
        }

        for (size_t frameIndex = 0u; frameIndex < FRAME_BUFFER_COUNT; ++frameIndex) {
            ID3D12CommandAllocator*& commandAllocator = this->m_commandAllocators[frameIndex];
            ID3D12GraphicsCommandList*& commandList = this->m_commandLists[frameIndex];

            // Create Command Allocator
            if (this->m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE::D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)) != S_OK)
                std::cout << "[DIRECTX 12] Failed To CreateCommandAllocator";

            // Create Command List
            if (this->m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator, NULL, IID_PPV_ARGS(&commandList)) != S_OK)
                std::cout << "[DIRECTX 12] Failed To CreateCommandList";
        }

        // Create Fences
        for (size_t frameIndex = 0u; frameIndex < FRAME_BUFFER_COUNT; ++frameIndex) {
            ID3D12Fence*& fence = this->m_fences[frameIndex];
            this->m_device->CreateFence(0, D3D12_FENCE_FLAGS::D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        }

        this->m_fenceEvent = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    }

    ~PolarTracer() noexcept {
        CloseHandle(this->m_fenceEvent);
    }

    void Run() noexcept {
        while (window.IsRunning()) {
            window.Update();

            const size_t frameIndex = this->m_swapChain->GetCurrentBackBufferIndex();
            
            this->m_commandLists[frameIndex]->Reset(this->m_commandAllocators[frameIndex], nullptr);
            
            D3D12_RESOURCE_BARRIER backBufferTransition;
            backBufferTransition.Type  = D3D12_RESOURCE_BARRIER_TYPE::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            backBufferTransition.Flags = D3D12_RESOURCE_BARRIER_FLAGS::D3D12_RESOURCE_BARRIER_FLAG_END_ONLY; // not sure
            backBufferTransition.Transition.pResource   = this->m_frameBuffers[frameIndex];
            backBufferTransition.Transition.Subresource = 0u;
            backBufferTransition.Transition.StateBefore = D3D12_RESOURCE_STATES::D3D12_RESOURCE_STATE_PRESENT;
            backBufferTransition.Transition.StateAfter  = D3D12_RESOURCE_STATES::D3D12_RESOURCE_STATE_RENDER_TARGET;
            
            this->m_commandLists[frameIndex]->ResourceBarrier(1u, &backBufferTransition);
            
            std::swap(backBufferTransition.Transition.StateBefore, backBufferTransition.Transition.StateAfter);
            this->m_commandLists[frameIndex]->ResourceBarrier(1u, &backBufferTransition);
            
            this->m_commandLists[frameIndex]->Close();
            this->m_commandQueue->ExecuteCommandLists(1u, reinterpret_cast<ID3D12CommandList* const*>(&this->m_commandLists[frameIndex]));
            
            const UINT fenceSignalValue = this->m_fenceValues[frameIndex] + 1u;
            this->m_commandQueue->Signal(this->m_fences[frameIndex], fenceSignalValue);
            
            while (this->m_fences[frameIndex]->GetCompletedValue() < fenceSignalValue) {
                this->m_fences[frameIndex]->SetEventOnCompletion(fenceSignalValue, this->m_fenceEvent);
                WaitForSingleObject(this->m_fenceEvent, INFINITE);
            }
            
            this->m_swapChain->Present(DXGI_SWAP_EFFECT_DISCARD, 0);
            
            ++this->m_fenceValues[frameIndex];
        }
    }

    //inline void RayTraceScene() {
       // const size_t bufferSize = outSurface.GetPixelCount() * sizeof(Math::Coloru8);
       //
       // // Allocate 1 thread per pixel of coordinates (X,Y). Use as many blocks in the grid as needed
       // // The RayTrace function will use the thread's index (both in the grid and in a block) to determine the pixel it will trace rays through
       // const dim3 dimBlock = dim3(16, 16); // Was 32x32: 32 warps of 32 threads per block (=1024 threads in total which is the hardware limit)
       // const dim3 dimGrid = dim3(std::ceil(this->host.m_renderParams.width / static_cast<float>(dimBlock.x)),
       //     std::ceil(this->host.m_renderParams.height / static_cast<float>(dimBlock.y)));
       //
       // // trace rays through each pixel
       // RayTracingDispatcher<<<dimGrid, dimBlock>>>(this->device.m_frameBuffer.GetPtr(), this->device.m_pRenderParams, this->device.m_primitives);
       //
       // // wait for the job to finish
       // printf("Job Finished with %s\n", cudaGetErrorString(cudaDeviceSynchronize()));


   // }
}; // PolarTracer

Memory::CPU_Array<Triangle> LoadObjectFile(const char* filename, const Material& material) {
    FILE* fp = fopen(filename, "r");

    if (!fp)
        printf("Error opening .obj File\n");

    Memory::CPU_Array<Math::Vec4f32> vertices;
    Memory::CPU_Array<Triangle> triangles;

    char lineBuffer[255];
    while (std::fgets(lineBuffer, sizeof(lineBuffer), fp) != nullptr) {
        switch (lineBuffer[0]) {
        case 'v':
        {
            std::istringstream lineStream(lineBuffer + 1);
            Math::Vec4f32 v;
            lineStream >> v.x >> v.y >> v.z;
            vertices.Append(v);
            break;
        }
        case 'f':
        {
            size_t vIndex0, vIndex1, vIndex2;
            std::istringstream lineStream(lineBuffer + 1);
            lineStream >> vIndex0 >> vIndex1 >> vIndex2;
            Triangle tr;
            tr.v0 = vertices[vIndex0 - 1];
            tr.v1 = vertices[vIndex1 - 1];
            tr.v2 = vertices[vIndex2 - 1];
            tr.material = material;

            triangles.Append(tr);
            break;
        }
        }
    }

    fclose(fp);

    return triangles;
}

#define WIDTH  (1920 / 4)
#define HEIGHT (1080 / 4)

int main(int argc, char** argv) {
    RenderParams renderParams;

    renderParams.width  = WIDTH;
    renderParams.height = HEIGHT;
    renderParams.camera.position = Math::Vec4f32(0.f, .0f, -2.f, 0.f);
    renderParams.camera.fov      = 3.141592f / 4.f;

    Primitives<Memory::Array, Device::CPU> primitives;
    primitives.spheres = Memory::CPU_Array<Sphere>(0);
    primitives.planes  = Memory::CPU_Array<Plane>(5);
    primitives.triangles = Memory::CPU_Array<Triangle>(0);

    for (auto& o : primitives.spheres) {
        o.material.reflectance = 0.f;
        o.material.roughness = 1.0f;
        o.material.transparency = 0.f;
        o.material.index_of_refraction = 1.0f;
    }

    for (auto& o : primitives.planes) {
        o.material.reflectance = 0.f;
        o.material.roughness = 1.0f;
        o.material.transparency = 0.f;
        o.material.index_of_refraction = 1.0f;
    }


    //primitives.spheres[1].center = Vec4f32{ 0.0f, 0.3f, 1.0f, 0.f };
    //primitives.spheres[1].radius = 0.5f;
    //primitives.spheres[1].material.diffuse   = Colorf32{ 1.f, 0.6f, 0.3f, 1.f };
    //primitives.spheres[1].material.emittance = Colorf32{ 0.f, 0.f, 0.f, 1.f };
    //primitives.spheres[1].material.reflectance = 0.1f;

    primitives.planes[0].position = Math::Vec4f32{ 0.f, -.25f, 0.f, 0.f};
    primitives.planes[0].normal   = Math::Vec4f32{ 0.f, 1.f, 0.f, 0.f};
    primitives.planes[0].material.diffuse   = Math::Colorf32{1.f, 1.f, 1.f, 1.f};
    primitives.planes[0].material.emittance = Math::Colorf32{0.25f, 0.25f, 0.25f, 1.f};

    primitives.planes[1].position = Math::Vec4f32{ 0.f, 0.f, 1.f, 0.f};
    primitives.planes[1].normal   = Math::Vec4f32{ 0.f, 0.f, -1.f, 0.f};
    primitives.planes[1].material.diffuse   = Math::Colorf32{0.75f, 0.75f, 0.75f, 1.f};
    primitives.planes[1].material.emittance = Math::Colorf32{0.f, 0.f, 0.f, 1.f};
    primitives.planes[1].material.reflectance = 1.f;
    primitives.planes[1].material.roughness   = 0.f;

    primitives.planes[2] = primitives.planes[1];
    primitives.planes[2].position = Math::Vec4f32{1.2f, 0.f, 0.f, 0.f};
    primitives.planes[2].normal   = Math::Vec4f32{-1.f, 0.f, 0.f, 0.f};
    primitives.planes[2].material.diffuse = {1.f, 0.f, 0.f, 1.f};
    primitives.planes[2].material.roughness   = 1.0f;
    primitives.planes[2].material.reflectance = 0.0f;

    primitives.planes[3] = primitives.planes[1];
    primitives.planes[3].position = Math::Vec4f32{-1.2f, 0.f, 0.f, 0.f};
    primitives.planes[3].normal   = Math::Vec4f32{1.f, 0.f, 0.f, 0.f};
    primitives.planes[3].material.roughness   = 0.0f;
    primitives.planes[3].material.reflectance = 1.0f;

    primitives.planes[4].position = Math::Vec4f32{ 0.f, 0.f, renderParams.camera.position.z - 1.f, 0.f};
    primitives.planes[4].normal   = Math::Vec4f32{ 0.f, 0.f, 1.f, 0.f};
    primitives.planes[4].material.diffuse   = Math::Colorf32{.75f, .75f, .75f, 1.f};
    primitives.planes[4].material.emittance = Math::Colorf32{0.25f, 0.25f, 0.25f, 1.f};

    Material bunnyMaterial;
    bunnyMaterial.diffuse   = Math::Colorf32{.75f, .75f, .75f, 1.f};
    bunnyMaterial.emittance = Math::Colorf32{0.f, 0.f, 0.f, 1.f};
    bunnyMaterial.reflectance = 0.f;
    bunnyMaterial.roughness = 1.0f;
    bunnyMaterial.transparency = 0.0f;
    bunnyMaterial.index_of_refraction = 1.0f;

   //auto bunnyTriangles = LoadObjectFile("res/bunny.obj", bunnyMaterial);
   //for (auto tr : bunnyTriangles) {
   //    tr.v0 *= 10; tr.v0.y -= 0.5f; tr.v0.z *= -1.f; tr.v0.x *= -1.f; tr.v0.x -= 0.1f;
   //    tr.v1 *= 10; tr.v1.y -= 0.5f; tr.v1.z *= -1.f; tr.v1.x *= -1.f; tr.v1.x -= 0.1f;
   //    tr.v2 *= 10; tr.v2.y -= 0.5f; tr.v2.z *= -1.f; tr.v2.x *= -1.f; tr.v2.x -= 0.1f;
   //    primitives.triangles.Append(tr);
   //}

    Memory::CPU_Array<CPU_Image<Math::Coloru8>> textures;
    PolarTracer pt(renderParams, primitives, textures);
    pt.Run();
}
