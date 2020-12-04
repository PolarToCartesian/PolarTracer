#include "Memory.cuh"

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