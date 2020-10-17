#pragma once

#include <cstdio>
#include <string>
#include <memory>
#include <cstdint>

#ifdef _WIN32
    #include <wrl.h>
    #include <wincodec.h>

    #define THROW_FATAL_ERROR(msg) MessageBoxA(NULL, msg, "Error", MB_ICONERROR)
#else
    #define THROW_FATAL_ERROR(msg) std::cerr << msg << '\n'
#endif

struct Coloru8 {
    std::uint8_t r, g, b, a;
}; // Coloru8

struct Colorf32 {
    float r, g, b, a;
}; // Colorf32

// Constants
class Image {
private:
    const std::uint16_t m_width;
    const std::uint16_t m_height;
    const std::uint32_t m_nPixels;

    std::unique_ptr<Coloru8[]> m_pBuff;

public:
    Image() = default;

    Image(const std::uint16_t width, const std::uint16_t height) noexcept;

    inline std::uint16_t GetWidth()      const noexcept { return this->m_width;   }
    inline std::uint16_t GetHeight()     const noexcept { return this->m_height;  }
    inline std::uint32_t GetPixelCount() const noexcept { return this->m_nPixels; }

    inline Coloru8* GetBufferPtr() const noexcept { return this->m_pBuff.get(); }

    inline       Coloru8& operator()(const size_t i)       noexcept { return this->m_pBuff[i]; }
    inline const Coloru8& operator()(const size_t i) const noexcept { return this->m_pBuff[i]; }

    inline       Coloru8& operator()(const size_t x, const size_t y)       noexcept { return this->m_pBuff[y * this->m_width + this->m_height]; }
    inline const Coloru8& operator()(const size_t x, const size_t y) const noexcept { return this->m_pBuff[y * this->m_width + this->m_height]; }

    void Save(const std::string& filename) noexcept;
}; // Image