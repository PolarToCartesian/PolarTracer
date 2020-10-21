#ifndef __PRTX__MEMORY_CU
#define __PRTX__MEMORY_CU

#include <string>

namespace PRTX {

    // The "Device" enum class represents the devices from which
    // memory can be accessed. This is necessary because the cpu can't
    // read/write directly from/to the GPU's memory and conversely.
    enum class Device { CPU, GPU }; // Device

    // The Pointer<typename _T, ::PRTX::Device _D> class represents a C++ Pointer of base type
    // _T that is accessible from the device _D (view enum ::PRTX::Device).
    template <typename _T, ::PRTX::Device _D>
    class Pointer {
    private:
        _T* m_raw = nullptr;
        
    public:
        __host__ __device__ inline Pointer() noexcept {  }

        __host__ __device__ inline Pointer(_T* const p)              noexcept { this->SetPointer(p); }
        __host__ __device__ inline Pointer(const Pointer<_T, _D>& o) noexcept { this->SetPointer(o); }

        __host__ __device__ inline void SetPointer(_T* const p)              noexcept { this->m_raw = p; }
        __host__ __device__ inline void SetPointer(const Pointer<_T, _D>& o) noexcept { this->SetPointer(o.m_raw); }

        __host__ __device__ inline _T*&       GetPointer()       noexcept { return this->m_raw; }
        __host__ __device__ inline _T* const& GetPointer() const noexcept { return this->m_raw; }

        template <typename _U>
        __host__ __device__ inline Pointer<_U, _D> AsPointerTo() const noexcept {
            return ::PRTX::Pointer<_U, _D>(reinterpret_cast<_U*>(this->m_raw));
        }

        __host__ __device__ inline void operator=(_T* const p)                      noexcept { this->SetPointer(p); }
        __host__ __device__ inline void operator=(const ::PRTX::Pointer<_T, _D>& o) noexcept { this->SetPointer(o); }

        __host__ __device__ inline operator _T*&      ()       noexcept { return this->m_raw; }
        __host__ __device__ inline operator _T* const&() const noexcept { return this->m_raw; }

        __host__ __device__ inline _T* const operator->() const noexcept { return this->m_raw; }
    }; // Pointer<_T>

    // Some aliases for the ::PRTX::Pointer<_T, _D> class.
    template <typename _T>
    using CPU_Ptr = ::PRTX::Pointer<_T, ::PRTX::Device::CPU>;

    template <typename _T>
    using GPU_Ptr = ::PRTX::Pointer<_T, ::PRTX::Device::GPU>;

    template <typename _T, ::PRTX::Device _D>
    __host__ __device__ inline ::PRTX::Pointer<_T, _D> AllocateSize(const size_t size) noexcept {
        if constexpr (_D == ::PRTX::Device::CPU) {
            return ::PRTX::CPU_Ptr<_T>(reinterpret_cast<_T*>(std::malloc(size)));
        } else {
            _T* p;
            cudaMalloc(reinterpret_cast<void**>(&p), size);
            return ::PRTX::GPU_Ptr<_T>(p);
        }
    }

    template <typename _T, ::PRTX::Device _D>
    __host__ __device__ inline ::PRTX::Pointer<_T, _D> AllocateCount(const size_t count) noexcept {
        return ::PRTX::AllocateSize<_T, _D>(count * sizeof(_T));
    }

    template <typename _T, ::PRTX::Device _D>
    __host__ __device__ inline ::PRTX::Pointer<_T, _D> AllocateSingle() noexcept {
        return ::PRTX::AllocateSize<_T, _D>(sizeof(_T));
    }

    template <typename _T, ::PRTX::Device _D>
    __host__ __device__ inline void Free(const ::PRTX::Pointer<_T, _D>& p) noexcept {
        if constexpr (_D == ::PRTX::Device::CPU) {
            std::free(p);
        } else {
            cudaFree(p.template AsPointerTo<void>());
        }
    }

    template <template<typename, ::PRTX::Device> typename _PTR_DST, typename _T_DST, ::PRTX::Device _D_DST,
              template<typename, ::PRTX::Device> typename _PTR_SRC, typename _T_SRC, ::PRTX::Device _D_SRC>
    __host__ __device__ inline void CopySize(const _PTR_DST<_T_DST, _D_DST>& dst,
                                             const _PTR_SRC<_T_SRC, _D_SRC>& src,
                                             const size_t size) noexcept
    {
        static_assert(std::is_same_v<_T_DST, _T_SRC>, "Incompatible Source And Destination Raw Pointer Types");
        
        cudaMemcpyKind memcpyKind;

        if constexpr (_D_SRC == ::PRTX::Device::CPU && _D_DST == ::PRTX::Device::CPU) {
            memcpyKind = cudaMemcpyKind::cudaMemcpyHostToHost;
        } else if constexpr (_D_SRC == ::PRTX::Device::GPU && _D_DST == ::PRTX::Device::GPU) {
            memcpyKind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
        } else if constexpr (_D_SRC == ::PRTX::Device::CPU && _D_DST == ::PRTX::Device::GPU) {
            memcpyKind = cudaMemcpyKind::cudaMemcpyHostToDevice;
        } else if constexpr (_D_SRC == ::PRTX::Device::GPU && _D_DST == ::PRTX::Device::CPU) {
            memcpyKind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
        } else { static_assert(1 == 1, "Incompatible Destination and Source Arguments"); }
   
        cudaMemcpy(dst, src, size, memcpyKind);
    }

    template <template<typename, ::PRTX::Device> typename _PTR_DST, typename _T_DST, ::PRTX::Device _D_DST,
              template<typename, ::PRTX::Device> typename _PTR_SRC, typename _T_SRC, ::PRTX::Device _D_SRC>
    __host__ __device__ inline void CopyCount(const _PTR_DST<_T_DST, _D_DST>& dst,
                                              const _PTR_SRC<_T_SRC, _D_SRC>& src,
                                              const size_t count) noexcept
    {
        static_assert(std::is_same_v<_T_DST, _T_SRC>, "Incompatible Source And Destination Raw Pointer Types");

        ::PRTX::CopySize(dst, src, count * sizeof(_T_DST));
    }

    template <template<typename, ::PRTX::Device> typename _PTR_DST, typename _T_DST, ::PRTX::Device _D_DST,
              template<typename, ::PRTX::Device> typename _PTR_SRC, typename _T_SRC, ::PRTX::Device _D_SRC>
    __host__ __device__ inline void CopySingle(const _PTR_DST<_T_DST, _D_DST>& dst,
                                               const _PTR_SRC<_T_SRC, _D_SRC>& src) noexcept
    {
        static_assert(std::is_same_v<_T_DST, _T_SRC>, "Incompatible Source And Destination Raw Pointer Types");
        
        ::PRTX::CopySize(dst, src, sizeof(_T_DST));
    }


    // The UniquePointer<typename _T, ::PRTX::Device _D> class represents a C++ Pointer
    // of base type _T whose memory is owned and managed by this class. As a result,
    // when this class is destroyed or it's owning memory location changes, it will
    // free the memory it owned.
    template <typename _T, ::PRTX::Device _D>
    class UniquePointer {
    private:
        ::PRTX::Pointer<_T, _D> m_ptr;
    
    public:
        __host__ __device__ inline UniquePointer() noexcept = default;
        
        __host__ __device__ inline UniquePointer(const ::PRTX::Pointer<_T, _D>& o)  noexcept {
            this->Free();
            this->m_ptr = o;
        }

        __host__ __device__ inline UniquePointer(::PRTX::UniquePointer<_T, _D>&& o) noexcept {
            this->Free();
            this->m_ptr = o.m_ptr;
            o.m_ptr     = nullptr;
        }

        __host__ __device__ inline void Free() const noexcept { ::PRTX::Free(this->m_ptr); }

        __host__ __device__ inline ~UniquePointer() noexcept { this->Free(); }

        __host__ __device__ inline ::PRTX::UniquePointer<_T, _D>& operator=(const ::PRTX::Pointer<_T, _D>& o)  noexcept {
            this->Free();
            this->m_ptr = o;
            return *this;
        }

        __host__ __device__ inline ::PRTX::UniquePointer<_T, _D>& operator=(::PRTX::UniquePointer<_T, _D>&& o) noexcept {
            this->Free();
            this->m_ptr = o;
            o.m_ptr     = nullptr;
            return *this;
        }

        __host__ __device__ inline const ::PRTX::Pointer<_T, _D>& GetPointer() const noexcept { return this->m_ptr; }

        __host__ __device__ inline operator const ::PRTX::Pointer<_T, _D>&() const noexcept { return this->m_ptr; }

        __host__ __device__ inline operator _T* const&() const noexcept { return this->m_ptr; }

        __host__ __device__ inline _T* const operator->() const noexcept { return this->m_ptr; }

        __host__ __device__ UniquePointer(const ::PRTX::UniquePointer<_T, _D>& o) = delete;
        __host__ __device__ ::PRTX::UniquePointer<_T, _D>& operator=(const ::PRTX::UniquePointer<_T, _D>& o) = delete;
    }; // UniquePointer<_T, _D>

    // Some aliases for the ::PRTX::UniquePointer<_T, _D> class.
    template <typename _T>
    using CPU_UniquePtr = ::PRTX::UniquePointer<_T, ::PRTX::Device::CPU>;

    template <typename _T>
    using GPU_UniquePtr = ::PRTX::UniquePointer<_T, ::PRTX::Device::GPU>;

    // The ArrayView<typename _T, ::PRTX::Device _D> class represents a
    // contiguous allocation of memory on the device _D of elements of type
    // _T. It is defined by a starting memory address and a count of elements
    // of type _T following the address.
    template <typename _T, ::PRTX::Device _D>
    class ArrayView {
    private:
        size_t m_count = 0;
        ::PRTX::Pointer<_T, _D> m_pBegin;
    
    protected:
        __host__ __device__ inline void SetPointer(const ::PRTX::Pointer<_T, _D>& pBegin) noexcept { this->m_pBegin = pBegin; }
        __host__ __device__ inline void SetCount  (const size_t count)                    noexcept { this->m_count  = count;  }

        __host__ __device__ inline operator _T*& () noexcept { return this->m_pBegin; }

    public:
        __host__ __device__ inline ArrayView() noexcept = default;

        __host__ __device__ inline ArrayView(const ::PRTX::Pointer<_T, _D>& pBegin, const size_t count) noexcept
            : m_pBegin(pBegin), m_count(count)
        {  }

        __host__ __device__ inline const ::PRTX::Pointer<_T, _D>& GetPointer() const noexcept { return this->m_pBegin; }
        __host__ __device__ inline const size_t&                  GetCount()   const noexcept { return this->m_count;  }

        __host__ __device__ inline operator const ::PRTX::Pointer<_T, _D>&() const noexcept { return this->m_pBegin; }

        __host__ __device__ inline operator _T* const&() const noexcept { return this->m_pBegin; }

        __host__ __device__ inline       _T& operator[](const size_t i)       noexcept { return *(this->m_pBegin + i); }
        __host__ __device__ inline const _T& operator[](const size_t i) const noexcept { return *(this->m_pBegin + i); }

        ~ArrayView() = default;
    }; // ArrayView<_T, _D>

    // Some aliases for the ::PRTX::ArrayView<_T, _D> class.
    template <typename _T>
    using CPU_ArrayView = ArrayView<_T, ::PRTX::Device::CPU>;

    template <typename _T>
    using GPU_ArrayView = ArrayView<_T, ::PRTX::Device::GPU>;


    // The Array<typename _T, ::PRTX::Device _D> is essentialy a
    // ArrayView<_T, _D> who owns the memory it represents.
    template <typename _T, ::PRTX::Device _D>
    class Array : public ArrayView<_T, _D> {
    public:
        __host__ __device__ inline Array() noexcept = default;

        __host__ __device__ inline Array(const size_t count) noexcept {
            this->SetCount(count);
            this->SetPointer(::PRTX::AllocateCount<_T, _D>(count));
        }
        
        template <::PRTX::Device _D_O>
        __host__ __device__ inline Array(const Array<_T, _D_O>& o) noexcept
          : Array(o.GetCount())
        {
            ::PRTX::CopyCount(*this, o, this->GetCount());
        }

        __host__ __device__ inline Array(Array<_T, _D>&& o) noexcept
        {
            this->SetCount(o.GetCount());
            o.SetCount(0);
            this->SetPointer(o.GetPointer());
            o.SetPointer((_T*)nullptr);
        }

        __host__ __device__ inline Array<_T, _D>& operator=(Array<_T, _D>&& o) noexcept {
            this->SetCount(o.GetCount());
            o.SetCount(0);
            this->SetPointer(o.GetPointer());
            o.SetPointer((_T*)nullptr);

            return *this;
        }

        __host__ __device__ inline void Reserve(const size_t count) noexcept {
            const auto newCount = this->GetCount() + count;
            const auto newBegin = ::PRTX::AllocateCount<_T, _D>(newCount);

            ::PRTX::CopyCount(newBegin, this->GetPointer(), this->GetCount());
            ::PRTX::Free(this->GetPointer());

            this->SetPointer(newBegin);
            this->SetCount(newCount);
        }

        __host__ __device__ inline ~Array() noexcept {
            ::PRTX::Free(this->GetPointer());
        }
    }; // Array<_T, _D>

    // Some aliases for the ::PRTX::Array<_T, _D> class.
    template <typename _T>
    using CPU_Array = Array<_T, ::PRTX::Device::CPU>;

    template <typename _T>
    using GPU_Array = Array<_T, ::PRTX::Device::GPU>;

}; // PRTX

#endif // __PRTX__MEMORY_CU