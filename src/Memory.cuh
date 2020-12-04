#ifndef __POLARTRACER__FILE_MEMORY_CUH
#define __POLARTRACER__FILE_MEMORY_CUH

#include "Pch.cuh"
#include "Common.cuh"

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

#endif // __POLARTRACER__FILE_MEMORY_CUH