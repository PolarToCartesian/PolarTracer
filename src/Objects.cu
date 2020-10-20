#ifndef __PRTX__OBJECTS_CU
#define __PRTX__OBJECTS_CU

#include "Math.cu"

namespace PRTX {

    struct Material {
        Colorf32 diffuse;
        Colorf32 emittance;
    }; // Material
    
    struct Sphere {
        Vec4f32 position;
        float   radius;
    
        Material material;
    }; // Sphere

}; // namespace PRTX

#endif // __PRTX__OBJECTS_CU