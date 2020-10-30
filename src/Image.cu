#ifndef __POLAR_TRACER__IMAGE_CU
#define __POLAR_TRACER__IMAGE_CU

#include "Vector.cu"
#include "Memory.cu"

template <typename T, Device D>
class Image {
private:
    std::uint16_t m_width   = 0;
    std::uint16_t m_height  = 0;
    std::uint32_t m_nPixels = 0;

    Array<T, D> m_pArray;

public:
    Image() = default;

    inline Image(const std::uint16_t width, const std::uint16_t height) noexcept
        : m_width(width),
        m_height(height),
        m_nPixels(static_cast<std::uint32_t>(width)* height),
        m_pArray(Array<T, D>(this->m_nPixels))
    { }

    __host__ __device__ inline std::uint16_t GetWidth()      const noexcept { return this->m_width; }
    __host__ __device__ inline std::uint16_t GetHeight()     const noexcept { return this->m_height; }
    __host__ __device__ inline std::uint32_t GetPixelCount() const noexcept { return this->m_nPixels; }

    __host__ __device__ inline Pointer<T, D> GetPtr() const noexcept { return this->m_pArray; }

    __host__ __device__ inline       T& operator()(const size_t i)       noexcept { return this->m_pArray[i]; }
    __host__ __device__ inline const T& operator()(const size_t i) const noexcept { return this->m_pArray[i]; }

    __host__ __device__ inline       T& operator()(const size_t x, const size_t y)       noexcept { return this->m_pArray[y * this->m_width + this->m_height]; }
    __host__ __device__ inline const T& operator()(const size_t x, const size_t y) const noexcept { return this->m_pArray[y * this->m_width + this->m_height]; }

}; // Image

void SaveImage(const Image<Coloru8, Device::CPU>& image, const std::string& filename) noexcept {
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

#endif // __POLAR_TRACER__IMAGE_CU