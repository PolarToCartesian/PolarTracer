#include "Pch.cuh"
#include "Memory.cu"
#include "Math.cu"
#include "Image.cu"
#include "Geometry.cu"
#include "Common.cuh"
#include "Camera.cuh"

struct RenderParams {
    size_t width  = 0;
    size_t height = 0;

    Camera camera;
};

// +-------+
// | Image |
// +-------+

CPU_Image<Math::Coloru8> ReadImage(const std::string& filename) noexcept {
    std::ifstream file(filename.c_str(), std::ifstream::binary);

	if (file.is_open()) {
		// Read Header
		std::string magicNumber;
		float maxColorValue;

        std::uint16_t width, height;
		file >> magicNumber >> width >> height >> maxColorValue;

		file.ignore(1); // Skip the last whitespace
        
        CPU_Image<Math::Coloru8> image(width, height);

		// Parse Image Data
        if (magicNumber == "P6") {
            const size_t bufferSize = image.GetPixelCount() * sizeof(Math::Coloru8);

            const Memory::CPU_UniquePtr<std::uint8_t> rawImageData = Memory::AllocateSize<std::uint8_t, Device::CPU>(bufferSize);
            file.read(rawImageData.AsPointerTo<char>(), bufferSize);

            size_t j = 0;
            for (size_t i = 0; i < image.GetPixelCount(); ++i)
                image(i) = Math::Coloru8(rawImageData[j++], rawImageData[j++], rawImageData[j++], 255u);
        } else {
			std::cout << magicNumber << " is not supported\n";
        }

        file.close();

        return image;
	} else {
		perror("Error while reading image file");
    }

    return CPU_Image<Math::Coloru8>(0, 0);
}

void SaveImage(const Image<Math::Coloru8, Device::CPU>& image, const std::string& filename) noexcept {
    const std::string fullFilename = filename + ".pam";

    // Open
    FILE* fp = std::fopen(fullFilename.c_str(), "wb");

    if (fp) {
        // Header
        std::fprintf(fp, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", image.GetWidth(), image.GetHeight());

        // Write Contents
        std::fwrite(image.GetPtr(), image.GetPixelCount() * sizeof(Math::Coloru8), 1u, fp);

        // Close
        std::fclose(fp);
    }
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
            
            const float rngd = Utility::RandomFloat(randState);

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
    const size_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

    // Bounds check
    if (pixelX >= pParams->width || pixelY >= pParams->height) return;

    curandState_t randState;
    curand_init(pixelX, pixelY, 0, &randState);

    const Ray cameraRay = pParams->camera.GenerateCameraRay(pixelX, pixelY, pParams->width, pParams->height, &randState);

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

class PolarTracer {
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

public:
    PolarTracer(const RenderParams& renderParams, const Primitives<Memory::Array, Device::CPU>& primitives = {}, const Memory::CPU_Array<CPU_Image<Math::Coloru8>>& textures = {})
        : host{ renderParams }
    {
        this->device.m_frameBuffer   = Image<Math::Coloru8, Device::GPU>(renderParams.width, renderParams.height);
        this->device.m_pRenderParams = Memory::AllocateSingle<RenderParams, Device::GPU>();
        this->device.m_primitives    = primitives;
        this->device.m_textures      = textures;

        const auto src = Memory::CPU_Ptr<RenderParams>(&this->host.m_renderParams);
        Memory::CopySingle(this->device.m_pRenderParams, src);
    }

    inline void RayTraceScene(const Image<Math::Coloru8, Device::CPU>& outSurface) {
        assert(outSurface.GetWidth() == this->host.m_renderParams.width && outSurface.GetHeight() == this->host.m_renderParams.height);

        const size_t bufferSize = outSurface.GetPixelCount() * sizeof(Math::Coloru8);

        // Allocate 1 thread per pixel of coordinates (X,Y). Use as many blocks in the grid as needed
        // The RayTrace function will use the thread's index (both in the grid and in a block) to determine the pixel it will trace rays through
        const dim3 dimBlock = dim3(16, 16); // Was 32x32: 32 warps of 32 threads per block (=1024 threads in total which is the hardware limit)
        const dim3 dimGrid = dim3(std::ceil(this->host.m_renderParams.width / static_cast<float>(dimBlock.x)),
            std::ceil(this->host.m_renderParams.height / static_cast<float>(dimBlock.y)));

        // trace rays through each pixel
        RayTracingDispatcher<<<dimGrid, dimBlock>>>(this->device.m_frameBuffer.GetPtr(), this->device.m_pRenderParams, this->device.m_primitives);

        // wait for the job to finish
        printf("Job Finished with %s\n", cudaGetErrorString(cudaDeviceSynchronize()));

        // copy the gpu buffer to a new cpu buffer
        Memory::CopySize(outSurface.GetPtr(), this->device.m_frameBuffer.GetPtr(), bufferSize);
    }
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