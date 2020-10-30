#include <cfloat>
#include <chrono>
#include <limits>
#include <ostream>
#include <cassert>
#include <stdio.h>
#include <optional>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Util.cu"
#include "Random.cu"
#include "Memory.cu"
#include "Vector.cu"
#include "Image.cu"

//TODO:
// + Check that the intersection normals always point in the correct direction
// + Fresnel
// + Obj Mesh loading

#define EPSILON (float)0.0001f
#define MAX_REC (10)
#define SPP     (10)

struct Camera {
    float   fov;
    Vec4f32 position;
}; // Camera

struct RenderParams {
    size_t width = 0;
    size_t height = 0;

    Camera camera;
};

struct Intersection;

struct Ray {
    Vec4f32 origin;
    Vec4f32 direction;

    template <typename _T_OBJ>
    __device__ Intersection Intersects(const _T_OBJ& obj) const noexcept;
}; // Ray

struct Material {
    Colorf32 diffuse   = { 0.f, 0.f, 0.f, 1.f };
    Colorf32 emittance = { 0.f, 0.f, 0.f, 1.f };

    float reflectance  = 0.f; // the sum of these values should be less than or equal to 1
    float transparency = 0.f; // the sum of these values should be less than or equal to 1

    float roughness = 0.f;

    float index_of_refraction = 1.f;
}; // Material

struct ObjectBase {
    Material material;
};

struct Sphere : ObjectBase {
    Vec4f32 center;
    float   radius;
}; // Sphere

struct Plane : ObjectBase {
    Vec4f32 position; // Any Point On The Plane
    Vec4f32 normal;   // Normal To The Surface
}; // Plane

struct Triangle : ObjectBase {
    Vec4f32 p0; // Position of the 1st vertex
    Vec4f32 p1; // Position of the 2nd vertex
    Vec4f32 p2; // Position of the 3rd vertex
};

struct Intersection {
    Ray      inRay;     // incoming ray
    float            t; // distance from the ray's origin to intersection point
    Vec4f32  location;  // intersection location
    Vec4f32  normal;    // normal at intersection point
    Material material;  // the material that the intersected object is made of

    __device__ __host__ static inline Intersection MakeNullIntersection(const Ray& ray) noexcept {
        return Intersection{ray, FLT_MAX};
    }
}; // Intersection

template <>
__device__ Intersection Ray::Intersects(const Sphere& sphere) const noexcept {
    const float radius2 = sphere.radius * sphere.radius;

    const Vec4f32 L = sphere.center - this->origin;
    const float   tca = Vec4f32::DotProduct3D(L, this->direction);
    const float   d2 = Vec4f32::DotProduct3D(L, L) - tca * tca;

    if (d2 > radius2)
        return Intersection::MakeNullIntersection(*this);

    const float thc = sqrt(radius2 - d2);
    float t0 = tca - thc;
    float t1 = tca + thc;

    if (t0 > t1) {
        const float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }

    if (t0 < EPSILON) {
        t0 = t1;

        if (t0 < 0)
            return Intersection::MakeNullIntersection(*this);
    }

    Intersection intersection;
    intersection.inRay    = *this;
    intersection.t        = t0;
    intersection.location = this->origin + t0 * this->direction;
    intersection.normal   = Vec4f32::Normalized3D(intersection.location - sphere.center);
    intersection.material = sphere.material;

    return intersection;
}

template <>
__device__ Intersection Ray::Intersects(const Plane& plane) const noexcept {
    const float denom = Vec4f32::DotProduct3D(plane.normal, this->direction);
    if (abs(denom) >= EPSILON) {
        const Vec4f32 v = plane.position - this->origin;
        const float t = Vec4f32::DotProduct3D(v, plane.normal) / denom;

        if (t >= 0) {
            Intersection intersection;
            intersection.inRay    = *this;
            intersection.t        = t;
            intersection.location = this->origin + t * this->direction;
            intersection.normal   = plane.normal;
            intersection.material = plane.material;

            return intersection;
        }
    }

    return Intersection::MakeNullIntersection(*this);
}

// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm#:~:text=The%20M%C3%B6ller%E2%80%93Trumbore%20ray%2Dtriangle,the%20plane%20containing%20the%20triangle.
template <>
__device__ Intersection Ray::Intersects(const Triangle& triangle) const noexcept {
    // compute plane's normal
    Vec4f32 v0v1 = triangle.p1 - triangle.p0; 
    Vec4f32 v0v2 = triangle.p2 - triangle.p0; 
    // no need to normalize
    Vec4f32 N = Vec4f32::CrossProduct3D(v0v1, v0v2); // N 
    float area2 = N.GetLength3D(); 
 
    // Step 1: finding P
 
    // check if ray and plane are parallel ?
    float NdotRayDirection = Vec4f32::DotProduct3D(N, this->direction); 
    if (fabs(NdotRayDirection) < EPSILON) // almost 0 
        return Intersection::MakeNullIntersection(*this); // they are parallel so they don't intersect ! 
 
    // compute d parameter using equation 2
    float d = Vec4f32::DotProduct3D(N, triangle.p0); 
 
    // compute t (equation 3)
    float t = (Vec4f32::DotProduct3D(N, this->origin) + d) / NdotRayDirection; 
    // check if the triangle is in behind the ray
    if (t < 0) return Intersection::MakeNullIntersection(*this); // the triangle is behind 
 
    // compute the intersection point using equation 1
    Vec4f32 P = this->origin + t * this->direction; 
 
    // Step 2: inside-outside test
    Vec4f32 C; // vector perpendicular to triangle's plane 
 
    // edge 0
    Vec4f32 edge0 = triangle.p1 - triangle.p0; 
    Vec4f32 vp0 = P - triangle.p0; 
    C = Vec4f32::DotProduct3D(edge0, vp0);
    if (Vec4f32::DotProduct3D(N, C) < 0) return Intersection::MakeNullIntersection(*this); // P is on the right side 
 
    // edge 1
    Vec4f32 edge1 = triangle.p2 - triangle.p1; 
    Vec4f32 vp1 = P - triangle.p1; 
    C = Vec4f32::DotProduct3D(edge1, vp1); 
    if (Vec4f32::DotProduct3D(N, C) < 0) return Intersection::MakeNullIntersection(*this);; // P is on the right side 
 
    // edge 2
    Vec4f32 edge2 = triangle.p0 - triangle.p2; 
    Vec4f32 vp2   = P - triangle.p2; 
    C = Vec4f32::DotProduct3D(edge2, vp2); 
    if (Vec4f32::DotProduct3D(N, C) < 0) return Intersection::MakeNullIntersection(*this);; // P is on the right side; 
 
    Intersection intersection;
    intersection.inRay    = *this;
    intersection.t        = t;
    intersection.location = P;
    intersection.normal   = N;//TODO:
    intersection.material = triangle.material;

    return intersection;
}

__device__ inline Ray GenerateCameraRay(const size_t& pixelX, const size_t& pixelY, const GPU_Ptr<RenderParams>& pRanderParams, curandState_t* const randState) noexcept {
    const RenderParams& renderParams = *pRanderParams;

    Ray ray;
    ray.origin = renderParams.camera.position;
    ray.direction = Vec4f32::Normalized3D(Vec4f32(
        (2.0f  * ((pixelX + RandomFloat(randState)) / static_cast<float>(renderParams.width))  - 1.0f) * tan(renderParams.camera.fov) * static_cast<float>(renderParams.width) / static_cast<float>(renderParams.height),
        (-2.0f * ((pixelY + RandomFloat(randState)) / static_cast<float>(renderParams.height)) + 1.0f) * tan(renderParams.camera.fov),
        1.0f,
        0.f));

    return ray;
}

template <typename T_OBJ>
__device__ inline void FindClosestIntersection(const Ray& ray,
                                               Intersection& closest, // in/out
                                               const GPU_ArrayView<T_OBJ>& objArrayView) noexcept {
    for (size_t i = 0; i < objArrayView.GetCount(); i++) {
        const Intersection current = ray.Intersects(objArrayView[i]);

        if (current.t < closest.t)
            closest = current;
    }
}

template <template<typename _IDC_T, Device _IDC_D> typename Container, Device D>
struct Primitives {
    Container<Sphere, D>   spheres;
    Container<Plane, D>    planes;
    Container<Triangle, D> triangles;

    Primitives() = default;

    template <Device D_2>
    inline Primitives(const Primitives<Container, D_2>& o) noexcept {
        this->spheres   = o.spheres;
        this->planes    = o.planes;
        this->triangles = o.triangles;
    }

    template <template<typename, Device> typename C_2, Device D_2>
    inline Primitives(const Primitives<C_2, D_2>& o) noexcept {
        this->spheres   = o.spheres;
        this->planes    = o.planes;
        this->triangles = o.triangles;
    }
}; // Primitives

__device__ Intersection
FindClosestIntersection(const Ray& ray, const Primitives<ArrayView, Device::GPU>& primitives) noexcept {
    Intersection closest;
    closest.t = FLT_MAX;
    
    FindClosestIntersection(ray, closest, primitives.spheres);
    FindClosestIntersection(ray, closest, primitives.planes);
    FindClosestIntersection(ray, closest, primitives.triangles);

    return closest;
}

template <size_t _N>
__device__ Colorf32 RayTrace(const Ray& ray,
                             const Primitives<ArrayView, Device::GPU>& primitives,
                             curandState_t* const randState) {
    const auto intersection = FindClosestIntersection(ray, primitives);

    if constexpr (_N < MAX_REC) {
        if (intersection.t != FLT_MAX) {
            const Material& material = intersection.material;
            
            Ray newRay;
            newRay.origin = intersection.location + EPSILON * intersection.normal;
            
            const float rngd = RandomFloat(randState);

            if (material.reflectance > rngd) {
                // Compute Reflexion
                newRay.direction = material.roughness * Random3DUnitVector(randState) + (1 - material.roughness) * Vec4f32::Reflected3D(ray.direction, intersection.normal);
            } else if (material.transparency + material.reflectance > rngd) {
                // Compute Transparency
                const bool outside = Vec4f32::DotProduct3D(ray.direction, intersection.normal) < 0;

                newRay.direction = Vec4f32::Normalized3D(Refract(ray.direction, intersection.normal, material.index_of_refraction));
                newRay.origin    = intersection.location + (outside ? -1 : 1) * EPSILON * intersection.normal;
            } else {
                // Compute Diffuse
                newRay.direction = Random3DUnitVector(randState);
            }
            
            const Colorf32 materialComp = RayTrace<_N + 1u>(newRay, primitives, randState);
            const Colorf32 finalColor   = material.emittance + material.diffuse * materialComp;

            return finalColor;
        }
    }
    
    // Black
    return Vec4f32{0.f, 0.f, 0.F, 1.f};
}

// Can't pass arguments via const& because these variables exist on the host and not on the device
__global__ void RayTracingDispatcher(const GPU_Ptr<Coloru8> pSurface,
                                     const GPU_Ptr<RenderParams> pParams,
                                     const Primitives<ArrayView, Device::GPU> primitives) {

    curandState_t randState;

    // Calculate the thread's (X, Y) location
    const size_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;

    curand_init(pixelX, pixelY, 0, &randState);

    // Bounds check
    if (pixelX >= pParams->width || pixelY >= pParams->height) return;

    // Determine the pixel's index into the image buffer
    const size_t index = pixelX + pixelY * pParams->width;

    const Ray cameraRay = GenerateCameraRay(pixelX, pixelY, pParams, &randState);

    // the current pixel's color (represented with floating point components)
    Colorf32 pixelColor{};
    for (size_t i = 0; i < SPP; i++)
        pixelColor += RayTrace<0>(cameraRay, primitives, &randState);
    
    pixelColor *= 255.f / static_cast<float>(SPP);
    pixelColor.Clamp(0.f, 255.f);

    // Save the result to the buffer
    *(pSurface + index) = Coloru8(pixelColor.x, pixelColor.y, pixelColor.z, pixelColor.w);
}

class PolarTracer {
private:
    struct {
        RenderParams m_renderParams;
    } host;

    struct {
        Image<Coloru8, Device::GPU> m_frameBuffer;
        GPU_UniquePtr<RenderParams> m_pRenderParams;

        Primitives<Array, Device::GPU> m_primitives;
    } device;

public:
    PolarTracer(const RenderParams& renderParams, const Primitives<Array, Device::CPU>& primitives)
        : host{ renderParams }
    {
        this->device.m_frameBuffer   = Image<Coloru8, Device::GPU>(renderParams.width, renderParams.height);
        this->device.m_pRenderParams = AllocateSingle<RenderParams, Device::GPU>();
        this->device.m_primitives    = primitives;

        const auto src = CPU_Ptr<RenderParams>(&this->host.m_renderParams);
        CopySingle(this->device.m_pRenderParams, src);
    }

    inline void RayTraceScene(const Image<Coloru8, Device::CPU>& outSurface) {
        assert(outSurface.GetWidth() == this->host.m_renderParams.width && outSurface.GetHeight() == this->host.m_renderParams.height);

        const size_t bufferSize = outSurface.GetPixelCount() * sizeof(Coloru8);

        // Allocate 1 thread per pixel of coordinates (X,Y). Use as many blocks in the grid as needed
        // The RayTrace function will use the thread's index (both in the grid and in a block) to determine the pixel it will trace rays through
        const dim3 dimBlock = dim3(16, 16); // Was 32x32: 32 warps of 32 threads per block (=1024 threads in total which is the hardware limit)
        const dim3 dimGrid = dim3(std::ceil(this->host.m_renderParams.width / static_cast<float>(dimBlock.x)),
            std::ceil(this->host.m_renderParams.height / static_cast<float>(dimBlock.y)));

        // trace rays through each pixel
        RayTracingDispatcher<<<dimGrid, dimBlock>>>(this->device.m_frameBuffer.GetPtr(),
            this->device.m_pRenderParams,
            this->device.m_primitives);

        // wait for the job to finish
        printf("%s\n", cudaGetErrorString(cudaDeviceSynchronize()));

        // copy the gpu buffer to a new cpu buffer
        CopySize(outSurface.GetPtr(), this->device.m_frameBuffer.GetPtr(), bufferSize);
    }
}; // PolarTracer

#define WIDTH  (1920)
#define HEIGHT (1080)

int main(int argc, char** argv) {
    Image<Coloru8, Device::CPU> image(WIDTH, HEIGHT);

    RenderParams renderParams;

    renderParams.width  = WIDTH;
    renderParams.height = HEIGHT;
    renderParams.camera.position = Vec4f32(0.f, .5f, -2.f, 0.f);
    renderParams.camera.fov      = 3.141592f / 4.f;

    Primitives<Array, Device::CPU> primitives;
    primitives.spheres = CPU_Array<Sphere>(2);
    primitives.planes  = CPU_Array<Plane>(5);

    for (auto& o : primitives.spheres) {
        o.material.reflectance = 0.f;
        o.material.roughness = 1.0f;
        o.material.transparency = 0.f;
        o.material.index_of_refraction = 1.0f;
    }

    for (auto& o : primitives.planes) {
        o.material.reflectance = 0.f;
        o.material.roughness = 1.0f;
        o.material.transparency = 0.f;
        o.material.index_of_refraction = 1.0f;
    }

    for (auto& o : primitives.triangles) {
        o.material.reflectance = 0.f;
        o.material.roughness = 1.0f;
        o.material.transparency = 0.f;
        o.material.index_of_refraction = 1.0f;
    }

    primitives.spheres[0].center = Vec4f32{ 0.0f, 1.5f, 0.5f, 0.f };
    primitives.spheres[0].radius = 0.5f;
    primitives.spheres[0].material.diffuse   = Colorf32{ 1.f, 1.f, 1.f, 1.f };
    const float li = 5.f;
    primitives.spheres[0].material.emittance = Colorf32{ li, li, li, 1.f };

    primitives.spheres[1].center = Vec4f32{ 0.0f, 0.3f, 1.0f, 0.f };
    primitives.spheres[1].radius = 0.5f;
    primitives.spheres[1].material.diffuse   = Colorf32{ 1.f, 0.6f, 0.3f, 1.f };
    primitives.spheres[1].material.emittance = Colorf32{ 0.f, 0.f, 0.f, 1.f };
    primitives.spheres[1].material.reflectance = 0.1f;

    primitives.planes[0].position = Vec4f32{ 0.f, -.25f, 0.f, 0.f};
    primitives.planes[0].normal   = Vec4f32{ 0.f, 1.f, 0.f, 0.f};
    primitives.planes[0].material.diffuse   = Colorf32{1.f, 1.f, 1.f, 1.f};
    primitives.planes[0].material.emittance = Colorf32{0.f, 0.f, 0.f, 1.f};

    primitives.planes[1].position = Vec4f32{ 0.f, 0.f, 1.f, 0.f};
    primitives.planes[1].normal   = Vec4f32{ 0.f, 0.f, -1.f, 0.f};
    primitives.planes[1].material.diffuse   = Colorf32{0.75f, 0.75f, 0.75f, 1.f};
    primitives.planes[1].material.emittance = Colorf32{0.f, 0.f, 0.f, 1.f};
    primitives.planes[1].material.reflectance = 1.f;
    primitives.planes[1].material.roughness   = 0.f;

    primitives.planes[2] = primitives.planes[1];
    primitives.planes[2].position = Vec4f32{1.f, 0.f, 0.f, 0.f};
    primitives.planes[2].normal   = Vec4f32{-1.f, 0.f, 0.f, 0.f};
    primitives.planes[2].material.roughness   = 0.f;
    primitives.planes[2].material.reflectance = 0.8f;

    primitives.planes[3] = primitives.planes[1];
    primitives.planes[3].position = Vec4f32{-1.f, 0.f, 0.f, 0.f};
    primitives.planes[3].normal   = Vec4f32{1.f, 0.f, 0.f, 0.f};
    primitives.planes[3].material.roughness = 0.25f;

    primitives.planes[4].position = Vec4f32{ 0.f, 0.f, renderParams.camera.position.z - 1.f, 0.f};
    primitives.planes[4].normal   = Vec4f32{ 0.f, 0.f, 1.f, 0.f};
    primitives.planes[4].material.diffuse   = Colorf32{.75f, .75f, .75f, 1.f};
    primitives.planes[4].material.emittance = Colorf32{0.f, 0.f, 0.f, 1.f};

    //float a = 0.25f;
    //primitives.triangles[0].p0 = Vec4f32{ -a,  +a,  2.f, 0.f};
    //primitives.triangles[0].p1 = Vec4f32{ +a,  +a,  2.f, 0.f};
    //primitives.triangles[0].p2 = Vec4f32{ 0.f, 0.f, 2.f, 0.f};
    //primitives.triangles[0].material.diffuse   = Colorf32{1.f, 1.f, 1.f, 1.f};
    //primitives.triangles[0].material.emittance = Colorf32{1.f, 1.f, 1.f, 1.f};

    PolarTracer pt(renderParams, primitives);

    const auto startTime = std::chrono::high_resolution_clock::now();
    pt.RayTraceScene(image);
    const auto endTime   = std::chrono::high_resolution_clock::now();

    const double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.f;

    std::cout << std::fixed << '\n';
    std::cout << "Took " << duration << "s to render a " << WIDTH << " by " << HEIGHT << " image at " << SPP << " SPP with a maximum recursion depth of " << MAX_REC << ".\n";
    std::cout << "Big numbers:\n";
    std::cout << "-->" << (unsigned int)(WIDTH * HEIGHT) << " pixels.\n";
    std::cout << "-->" << (unsigned int)(WIDTH * HEIGHT * SPP) << " samples.\n";
    std::cout << "-->" << (unsigned int)(WIDTH * HEIGHT * SPP * MAX_REC) << " photons.\n";
    std::cout << "Timings:\n";
    std::cout << "-->" << (unsigned int)((WIDTH * HEIGHT * SPP) / duration) << " samples per sec.\n";
    std::cout << "-->" << (unsigned int)((WIDTH * HEIGHT * SPP * MAX_REC) / duration) << " photon paths per sec.\n";

    SaveImage(image, "frame");
}
