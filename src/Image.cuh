#ifndef __POLARTRACER__FILE_IMAGE_CUH
#define __POLARTRACER__FILE_IMAGE_CUH

#include "Pch.cuh"
#include "Memory.cuh"
#include "Common.cuh"

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


#endif // __POLARTRACER__FILE_IMAGE_CUH