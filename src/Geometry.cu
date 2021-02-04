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

template<>
__device__ Intersection Ray::Intersects(const Triangle& triangle) const noexcept {
    Math::Vec4f32 edge1 = triangle.v1 - triangle.v0;
    Math::Vec4f32 edge2 = triangle.v2 - triangle.v0;
    Math::Vec4f32 h     = Math::CrossProduct3D(this->direction, edge2);

    const float a = Math::DotProduct3D(edge1, h);
    if (a > -EPSILON && a < EPSILON)
        return Intersection::MakeNullIntersection(*this);    // This ray is parallel to this triangle.
    
    const float f = 1.0f/a;

    const Math::Vec4f32 s = this->origin - triangle.v0;
    const float u = f * Math::DotProduct3D(s, h);
    if (u < 0.0 || u > 1.0)
        return Intersection::MakeNullIntersection(*this);
    
    const Math::Vec4f32 q = Math::CrossProduct3D(s, edge1);
    const float v = f * Math::DotProduct3D(this->direction, q);
    if (v < 0.0 || u + v > 1.0)
        return Intersection::MakeNullIntersection(*this);

    const float t = f * Math::DotProduct3D(edge2, q);
    if (t > EPSILON) {
        Intersection intersection;
        intersection.inRay    = *this;
        intersection.t        = t;
        intersection.location = this->origin + intersection.t * this->direction;
        intersection.normal   = Math::Normalized3D(Math::CrossProduct3D(edge1, edge2));
        intersection.material = triangle.material;

        return intersection;
    }
    
    return Intersection::MakeNullIntersection(*this);
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