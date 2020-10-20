#ifndef __PRTX__MEMORY_CU
#define __PRTX__MEMORY_CU

#include <string>

namespace PRTX {

    enum class Device { CPU, GPU }; // Device

    template <typename _T, ::PRTX::Device _D>
    class Pointer {
    private:
        _T* m_raw = nullptr;
        
    public:
        __host__ __device__ inline Pointer() noexcept {  }

        template <typename _U>
        __host__ __device__ inline Pointer(_U* const p) noexcept
          : m_raw(p)
        {
          static_assert(std::is_same<_T, _U>::value);
        }

        __host__ __device__ inline Pointer(const Pointer<_T, _D>& o) noexcept
          : m_raw(o.m_raw)
        {  }

        __host__ __device__ inline _T* Get() const noexcept { return this->m_raw; }

        template <typename _U = _T>
        __host__ __device__ inline void operator=(_U* const p) noexcept {
          this->m_raw = p; 
          static_assert(std::is_same<_T, _U>::value);
        }
        
        __host__ __device__ inline void operator=(const ::PRTX::Pointer<_T, _D>& o) noexcept { this->m_raw = o.m_raw; }

        __host__ __device__ inline operator _T*&      ()       noexcept { return this->m_raw; }
        __host__ __device__ inline operator _T* const&() const noexcept { return this->m_raw; }

        __host__ __device__ inline _T* operator->() const noexcept { return this->m_raw; }
    }; // Pointer<_T>

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
            cudaMalloc(&p, size);
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
            cudaFree(reinterpret_cast<void*>(p.Get()));
        }
    }

    template <typename _T, ::PRTX::Device _D_DST, ::PRTX::Device _D_SRC>
    __host__ __device__ inline void CopySize(const ::PRTX::Pointer<_T, _D_DST>& dst,
                                             const ::PRTX::Pointer<_T, _D_SRC>& src,
                                             const size_t size) noexcept
    {
        if constexpr (_D_SRC == ::PRTX::Device::CPU && _D_DST == ::PRTX::Device::CPU) {
            cudaMemcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyHostToHost);
        } else if constexpr (_D_SRC == ::PRTX::Device::GPU && _D_DST == ::PRTX::Device::GPU) {
            cudaMemcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        } else if constexpr (_D_SRC == ::PRTX::Device::CPU && _D_DST == ::PRTX::Device::GPU) {
            cudaMemcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
        } else if constexpr (_D_SRC == ::PRTX::Device::GPU && _D_DST == ::PRTX::Device::CPU) {
            cudaMemcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        } else { static_assert(1 == 1, "Incompatible Destination and Source Arguments"); }
    }

    template <typename _T, ::PRTX::Device _D_DST, ::PRTX::Device _D_SRC>
    __host__ __device__ inline void CopyCount(const ::PRTX::Pointer<_T, _D_DST>& dst,
                                              const ::PRTX::Pointer<_T, _D_SRC>& src,
                                              const size_t count) noexcept
    {
        ::PRTX::CopySize(dst, src, count * sizeof(_T));
    }

    template <typename _T, ::PRTX::Device _D_DST, ::PRTX::Device _D_SRC>
    __host__ __device__ inline void CopySingle(const ::PRTX::Pointer<_T, _D_DST>& dst,
                                               const ::PRTX::Pointer<_T, _D_SRC>& src) noexcept
    {
        ::PRTX::CopySize(dst, src, sizeof(_T));
    }

    template <typename _T, ::PRTX::Device _D>
    class ArraySpan {
    private:
        size_t m_count = 0;
        ::PRTX::Pointer<_T, _D> m_pBegin;
    
    protected:
        // These functions are protected so that users of the Array class can't use these functions
        // we want this because the Array class manages the memory it owns. The user should not be able
        // to change to the array's start pointer without the class freeing the memory it was owning
        __host__ __device__ inline void SetPtr  (const ::PRTX::Pointer<_T, _D>& pBegin) noexcept { this->m_pBegin = pBegin; }
        __host__ __device__ inline void SetCount(const size_t count)                    noexcept { this->m_count  = count;  }

    public:
        __host__ __device__ inline ArraySpan() noexcept = default;

        __host__ __device__ inline ArraySpan(const ::PRTX::Pointer<_T, _D>& pBegin, const size_t count) noexcept
            : m_pBegin(pBegin), m_count(count)
        {  }

        __host__ __device__ inline ::PRTX::Pointer<_T, _D> GetPtr()   const noexcept { return this->m_pBegin; }
        __host__ __device__ inline size_t                  GetCount() const noexcept { return this->m_count;  }

        __host__ __device__ inline operator ::PRTX::Pointer<_T, _D>() const noexcept { return this->m_pBegin; }

        __host__ __device__ inline       _T& operator[](const size_t i)       noexcept { return *(this->m_pBegin + i); }
        __host__ __device__ inline const _T& operator[](const size_t i) const noexcept { return *(this->m_pBegin + i); }

        ~ArraySpan() = default;
    };

    template <typename _T>
    using CPU_ArraySpan = ArraySpan<_T, ::PRTX::Device::CPU>;

    template <typename _T>
    using GPU_ArraySpan = ArraySpan<_T, ::PRTX::Device::GPU>;

    template <typename _T, ::PRTX::Device _D>
    class Array : public ArraySpan<_T, _D> {
    public:
        __host__ __device__ inline Array() noexcept = default;

        __host__ __device__ inline Array(const size_t count) noexcept {
            this->SetCount(count);
            this->SetPtr(::PRTX::AllocateCount<_T, _D>(count));
        }
        
        template <::PRTX::Device _D_O>
        __host__ __device__ inline Array(const Array<_T, _D_O>& o) noexcept
          : Array(o.GetCount())
        {
            ::PRTX::CopyCount(this->GetPtr(), o.GetPtr(), this->GetCount());
        }

        __host__ __device__ inline Array(Array<_T, _D>&& o) noexcept
        {
            this->SetCount(o.GetCount());
            o.SetCount(0);
            this->SetPtr(o.GetPtr());
            o.SetPtr((_T*)nullptr);
        }

        __host__ __device__ inline Array<_T, _D>& operator=(Array<_T, _D>&& o) noexcept {
            this->SetCount(o.GetCount());
            o.SetCount(0);
            this->SetPtr(o.GetPtr());
            o.SetPtr((_T*)nullptr);

            return *this;
        }

        __host__ __device__ inline void Reserve(const size_t count) noexcept {
            const auto newCount = this->GetCount() + count;
            const auto newBegin = ::PRTX::AllocateCount<_T, _D>(newCount);

            ::PRTX::CopyCount(newBegin, this->GetPtr(), this->GetCount());
            ::PRTX::Free(this->GetPtr());

            this->SetPtr(newBegin);
            this->SetCount(newCount);
        }

        __host__ __device__ inline ~Array() noexcept {
            ::PRTX::Free(this->GetPtr());
        }
    }; // Array<_T, _D>

    template <typename _T>
    using CPU_Array = Array<_T, ::PRTX::Device::CPU>;

    template <typename _T>
    using GPU_Array = Array<_T, ::PRTX::Device::GPU>;

}; // PRTX

#endif // __PRTX__MEMORY_CU