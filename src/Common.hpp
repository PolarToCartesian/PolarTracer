#ifndef __POLAR_TRACER__COMMON_HPP
#define __POLAR_TRACER__COMMON_HPP

#include <memory>
#include <string>
#include <cstdio>
#include <cstdint>

template <typename _T>
struct Color { _T r, g, b, a; };

typedef Color<float>        Colorf32;
typedef Color<std::uint8_t> Coloru8;

class Image {
private:
    const std::uint16_t m_width   = 0;
    const std::uint16_t m_height  = 0;
    const std::uint32_t m_nPixels = 0;

    std::unique_ptr<Coloru8[]> m_pBuff;

public:
    Image() = default;

    Image(const std::uint16_t width, const std::uint16_t height) noexcept 
      : m_width(width), m_height(height), m_nPixels(static_cast<std::uint32_t>(width) * height)
    {
        this->m_pBuff = std::make_unique<Coloru8[]>(this->m_nPixels);
    }

    inline std::uint16_t GetWidth()      const noexcept { return this->m_width;   }
    inline std::uint16_t GetHeight()     const noexcept { return this->m_height;  }
    inline std::uint32_t GetPixelCount() const noexcept { return this->m_nPixels; }

    inline Coloru8* GetBufferPtr() const noexcept { return this->m_pBuff.get(); }

    inline       Coloru8& operator()(const size_t i)       noexcept { return this->m_pBuff[i]; }
    inline const Coloru8& operator()(const size_t i) const noexcept { return this->m_pBuff[i]; }

    inline       Coloru8& operator()(const size_t x, const size_t y)       noexcept { return this->m_pBuff[y * this->m_width + this->m_height]; }
    inline const Coloru8& operator()(const size_t x, const size_t y) const noexcept { return this->m_pBuff[y * this->m_width + this->m_height]; }

    void Save(const std::string& filename) noexcept {
        const std::string fullFilename = filename + ".pam";
  
        // Open
        FILE* fp = std::fopen(fullFilename.c_str(), "wb");
  
        if (fp) {
            // Header
            std::fprintf(fp, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", this->m_width, this->m_height);
  
            // Write Contents
            std::fwrite(this->m_pBuff.get(), this->m_nPixels * sizeof(Coloru8), 1u, fp);
  
            // Close
            std::fclose(fp);
        }
    }

}; // Image

#endif // __POLAR_TRACER__COMMON_HPP