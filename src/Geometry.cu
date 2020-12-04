#include "Geometry.cuh"

template <>
__device__ Intersection Ray::Intersects(const Sphere& sphere) const noexcept {
    const float radius2 = sphere.radius * sphere.radius;

    const auto L   = sphere.center - this->origin;
    const auto tca = Math::DotProduct3D(L, this->direction);
    const auto d2  = Math::DotProduct3D(L, L) - tca * tca;

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
    intersection.normal   = Math::Normalized3D(intersection.location - sphere.center);
    intersection.material = sphere.material;

    return intersection;
}

template <>
__device__ Intersection Ray::Intersects(const Plane& plane) const noexcept {
    const float denom = Math::DotProduct3D(plane.normal, this->direction);
    if (abs(denom) >= EPSILON) {
        const auto  v = plane.position - this->origin;
        const float t = Math::DotProduct3D(v, plane.normal) / denom;

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

//https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle
template <>
__device__ Intersection Ray::Intersects(const Triangle& triangle) const noexcept {
    float u = 0.f, v = 0.f;

    const auto v0v1 = triangle.v1 - triangle.v0; 
    const auto v0v2 = triangle.v2 - triangle.v0; 
    const auto pvec = Math::CrossProduct3D(v0v2,  this->direction);

    const auto det = Math::DotProduct3D(v0v1, pvec); 

    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < EPSILON) return Intersection::MakeNullIntersection(*this);

    const float invDet = 1.f / det; 
 
    const auto tvec = this->origin - triangle.v0; 
    u = Math::DotProduct3D(tvec, pvec) * invDet; 
    if (u < 0 || u > 1) return Intersection::MakeNullIntersection(*this); 
 
    const auto qvec = Math::CrossProduct3D(tvec, v0v1); 
    v = Math::DotProduct3D(this->direction, qvec) * invDet; 
    if (v < 0 || u + v > 1) return Intersection::MakeNullIntersection(*this); 
 
    Intersection intersection;
    intersection.t        = Math::DotProduct3D(v0v2, qvec) * invDet;

    if (intersection.t <= 0) return Intersection::MakeNullIntersection(*this); 

    intersection.inRay    = *this;
    intersection.location = this->origin + intersection.t * this->direction;
    intersection.normal   = Math::Normalized3D(Math::CrossProduct3D(v0v1, v0v2)); //TODO:: Actual Normal
    intersection.material = triangle.material;

    return intersection;
}

template <typename T_OBJ>
__device__ inline void FindClosestIntersection(const Ray& ray,
                                               Intersection& closest, // in/out
                                               const Memory::GPU_ArrayView<T_OBJ>& objArrayView) noexcept {
    for (const T_OBJ& obj : objArrayView) {
        const Intersection current = ray.Intersects(obj);

        if (current < closest)
            closest = current;
    }
}

__device__ Intersection FindClosestIntersection(const Ray& ray, const Primitives<Memory::ArrayView, Device::GPU>& primitives) noexcept {
    Intersection closest;
    closest.t = FLT_MAX;
    
    FindClosestIntersection(ray, closest, primitives.spheres);
    FindClosestIntersection(ray, closest, primitives.planes);
    FindClosestIntersection(ray, closest, primitives.triangles);

    return closest;
}