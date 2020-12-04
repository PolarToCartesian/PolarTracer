#ifndef __POLARTRACER__FILE_GEOMETRY_CUH
#define __POLARTRACER__FILE_GEOMETRY_CUH

#include "Math.cuh"
#include "Memory.cuh"

struct Intersection;

struct Ray {
    Math::Vec4f32 origin;
    Math::Vec4f32 direction;

    template <typename _T_OBJ>
    __device__ Intersection Intersects(const _T_OBJ& obj) const noexcept;
}; // Ray

struct Material {
    Math::Colorf32 diffuse   = { 0.f, 0.f, 0.f, 1.f };
    Math::Colorf32 emittance = { 0.f, 0.f, 0.f, 1.f };

    float reflectance  = 0.f; // the sum of these values should be less than or equal to 1
    float transparency = 0.f; // the sum of these values should be less than or equal to 1

    float roughness = 0.f;

    float index_of_refraction = 1.f;
}; // Material

struct SphereGeometry {
    Math::Vec4f32 center;
    float         radius;
}; // SphereGeometry

struct Sphere : SphereGeometry {
    Material material;
}; // Sphere

struct PlaneGeometry {
    Math::Vec4f32 position; // Any Point On The Plane
    Math::Vec4f32 normal;   // Normal To The Surface
}; // PlaneGeometry

struct Plane : PlaneGeometry {
    Material material;
}; // Plane

struct TriangleGeometry {
    Math::Vec4f32 v0; // Position of the 1st vertex
    Math::Vec4f32 v1; // Position of the 2nd vertex
    Math::Vec4f32 v2; // Position of the 3rd vertex
}; // TriangleGeometry

struct Triangle : TriangleGeometry {
    Material material;
}; // Triangle

struct BoxGeometry {
    Math::Vec4f32 center;
    float         sideLength;
}; // BoxGeometry

struct Box : BoxGeometry {
    Material material;
}; // Box

struct Intersection {
    Ray           inRay;     // incoming ray
    float         t;         // distance from the ray's origin to intersection point
    Math::Vec4f32 location;  // intersection location
    Math::Vec4f32 normal;    // normal at intersection point
    Material material;       // the material that the intersected object is made of

    __device__ __host__ inline bool operator<(const Intersection& o) const noexcept {
    	return this->t < o.t;
    }

    __device__ __host__ inline bool operator>(const Intersection& o) const noexcept {
    	return this->t > o.t;
    }

    static __device__ __host__ inline Intersection MakeNullIntersection(const Ray& ray) noexcept {
        return Intersection{ray, FLT_MAX};
    }
}; // Intersection

// Shoud me much simpler with concepts (maybe :-)
template <template<typename, Device> typename Container, Device D>
struct Primitives {
    Container<Sphere,   D> spheres;
    Container<Plane,    D> planes;
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

template <typename T_OBJ>
__device__ inline void FindClosestIntersection(const Ray& ray,
                                               Intersection& closest, // in/out
                                               const Memory::GPU_ArrayView<T_OBJ>& objArrayView) noexcept;

__device__ Intersection FindClosestIntersection(const Ray& ray, const Primitives<Memory::ArrayView, Device::GPU>& primitives) noexcept;

#endif // __POLARTRACER__FILE_GEOMETRY_CUH