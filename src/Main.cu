#ifndef __PRTX__MAIN_CU
#define __PRTX__MAIN_CU

#include "PolarTracer.cu"

using namespace PRTX;

#define WIDTH  (1920)
#define HEIGHT (1080)

int main(int argc, char** argv) {    
    Image<Coloru8, Device::CPU> image(WIDTH, HEIGHT);

    ::PRTX::RenderParams renderParams;

    renderParams.width  = WIDTH;
    renderParams.height = HEIGHT;
    renderParams.camera.position = Vec4f32();
    renderParams.camera.fov      = M_PI / 4.f;

    CPU_Array<Sphere> spheres(1);
    spheres[0].position = Vec4f32{0.0f, 0.0f, 2.f, 0.f};
    spheres[0].radius   = 0.25f;
    spheres[0].material.diffuse   = Colorf32{1.f, 0.f, 1.f, 1.f};
    spheres[0].material.emittance = Colorf32{0.f, 0.f, 0.f, 0.f};
    
    PolarTracer pt(renderParams, spheres);
    pt.RayTraceScene(image);

    SaveImage(image, "frame");
}

#endif // __PRTX__MAIN_CU