#ifndef __PRTX__MAIN_CU
#define __PRTX__MAIN_CU

#include "PolarTracer.cu"

int main(int argc, char** argv) {
    const size_t WIDTH = 1920, HEIGHT = 1080;
    PRTX::Image image(WIDTH, HEIGHT);

    PRTX::Camera camera;
    camera.position = PRTX::Vec4f32();
    camera.fov      = M_PI / 4.f;

    PRTX::CPU_Array<PRTX::Sphere> spheres(1);
    spheres[0].position = PRTX::Vec4f32{0.0f, 0.0f, 2.f, 0.f};
    spheres[0].radius   = 0.25f;
    spheres[0].material.diffuse   = PRTX::Colorf32{1.f, 0.f, 1.f, 1.f};
    spheres[0].material.emittance = PRTX::Colorf32{0.f, 0.f, 0.f, 0.f};
    
    PRTX::PolarTracer pt(WIDTH, HEIGHT, camera, spheres);
    pt.RayTraceScene(image);

    image.Save("frame");
}

#endif // __PRTX__MAIN_CU