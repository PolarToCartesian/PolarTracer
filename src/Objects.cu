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

}; // namespace PRTX

#endif // __PRTX__OBJECTS_CU