#ifndef __PRTX__CAMERA_CU
#define __PRTX__CAMERA_CU

#include "Math.cu"

namespace PRTX {

    struct Camera {
        float   fov;
        ::PRTX::Vec4f32 position;
    }; // Camera

}; // namespace PRTX

#endif // __PRTX__CAMERA_CU