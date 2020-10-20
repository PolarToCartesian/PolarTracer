#ifndef __PRTX__OBJECTS_CU
#define __PRTX__OBJECTS_CU

#include "Math.cu"

namespace PRTX {

    struct Material {
        ::PRTX::Colorf32 diffuse;
        ::PRTX::Colorf32 emittance;
    }; // Material
    
    struct Sphere {
        ::PRTX::Vec4f32 position;
        float   radius;
    
        ::PRTX::Material material;
    }; // Sphere

    struct Intersection {
        ::PRTX::Ray      inRay;    // incoming ray
        float            t;        // distance from the ray's origin to intersection point
        ::PRTX::Ray      normal;   // normal at intersection point
        ::PRTX::Material material; // the material that the intersected object is made of      ::      
    }; // Intersection

}; // namespace PRTX

#endif // __PRTX__OBJECTS_CU