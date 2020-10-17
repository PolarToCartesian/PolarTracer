#include "Image.hpp"

Image::Image(const std::uint16_t width, const std::uint16_t height) noexcept
    : m_width(width), m_height(height), m_nPixels(static_cast<std::uint32_t>(width) * height)
{
    this->m_pBuff = std::make_unique<Coloru8[]>(this->m_nPixels);
}

void Image::Save(const std::string& filename) noexcept { // filename shouldn't contain the file extension
#ifdef _WIN32 // Use WIC (Windows Imaging Component) To Save A .PNG File Natively

    const std::string  fullFilename = filename + ".png";
    const std::wstring fullFilenameW(fullFilename.begin(), fullFilename.end());

    Microsoft::WRL::ComPtr<IWICImagingFactory>    factory;
    Microsoft::WRL::ComPtr<IWICBitmapEncoder>     bitmapEncoder;
    Microsoft::WRL::ComPtr<IWICBitmapFrameEncode> bitmapFrame;
    Microsoft::WRL::ComPtr<IWICStream>            outputStream;

    if (CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)) != S_OK)
        THROW_FATAL_ERROR("[WIC] Could Not Create IWICImagingFactory");

    if (factory->CreateStream(&outputStream) != S_OK)
        THROW_FATAL_ERROR("[WIC] Failed To Create Output Stream");

    if (outputStream->InitializeFromFilename(fullFilenameW.c_str(), GENERIC_WRITE) != S_OK)
        THROW_FATAL_ERROR("[WIC] Failed To Initialize Output Stream From Filename");

    if (factory->CreateEncoder(GUID_ContainerFormatPng, NULL, &bitmapEncoder) != S_OK)
        THROW_FATAL_ERROR("[WIC] Failed To Create Bitmap Encoder");

    if (bitmapEncoder->Initialize(outputStream.Get(), WICBitmapEncoderNoCache) != S_OK)
        THROW_FATAL_ERROR("[WIC] Failed To Initialize Bitmap ");

    if (bitmapEncoder->CreateNewFrame(&bitmapFrame, NULL) != S_OK)
        THROW_FATAL_ERROR("[WIC] Failed To Create A New Frame");

    if (bitmapFrame->Initialize(NULL) != S_OK)
        THROW_FATAL_ERROR("[WIC] Failed To Initialize A Bitmap's Frame");

    if (bitmapFrame->SetSize(this->m_width, this->m_height) != S_OK)
        THROW_FATAL_ERROR("[WIC] Failed To Set A Bitmap's Frame's Size");

    const WICPixelFormatGUID desiredPixelFormat = GUID_WICPixelFormat32bppBGRA;

    WICPixelFormatGUID currentPixelFormat = {};
    if (bitmapFrame->SetPixelFormat(&currentPixelFormat) != S_OK)
        THROW_FATAL_ERROR("[WIC] Failed To Set Pixel Format On A Bitmap Frame's");

    if (!IsEqualGUID(currentPixelFormat, desiredPixelFormat))
        THROW_FATAL_ERROR("[WIC] The Requested Pixel Format Is Not Supported");

    if (bitmapFrame->WritePixels(this->m_height, this->m_width * sizeof(Coloru8), this->m_nPixels * sizeof(Coloru8), (BYTE*)this->m_pBuff.get()) != S_OK)
        THROW_FATAL_ERROR("[WIC] Failed To Write Pixels To A Bitmap's Frame");

    if (bitmapFrame->Commit() != S_OK)
        THROW_FATAL_ERROR("[WIC] Failed To Commit A Bitmap's Frame");

    if (bitmapEncoder->Commit() != S_OK)
        THROW_FATAL_ERROR("[WIC] Failed To Commit Bitmap Encoder");

#else // On Other Operating Systems, Simply Write A .PAM File

    const std::string fullFilename = filename + ".pam";

    // Open
    FILE* fp = std::fopen(fullFilename.c_str(), "wb");

    if (fp) {
        // Header
        std::fprintf(fp, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\n MAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", this->m_width, this->m_height);

        // Write Contents
        std::fwrite(this->m_pBuff.get(), this->m_nPixels * sizeof(Coloru8), 1u, fp);

        // Close
        std::fclose(fp);
    }
}

#endif