#include "PolarTracer.cu"

#define WIDTH  (1920)
#define HEIGHT (1080)

int main(int argc, char** argv) {
    Image<Math::Coloru8, Device::CPU> image(WIDTH, HEIGHT);

    RenderParams renderParams;

    renderParams.width  = WIDTH;
    renderParams.height = HEIGHT;
    renderParams.camera.position = Math::Vec4f32(0.f, .0f, -2.f, 0.f);
    renderParams.camera.fov      = 3.141592f / 4.f;

    Primitives<Memory::Array, Device::CPU> primitives;
    primitives.spheres = Memory::CPU_Array<Sphere>(0);
    primitives.planes  = Memory::CPU_Array<Plane>(5);
    primitives.triangles = Memory::CPU_Array<Triangle>(0);

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


    //primitives.spheres[1].center = Vec4f32{ 0.0f, 0.3f, 1.0f, 0.f };
    //primitives.spheres[1].radius = 0.5f;
    //primitives.spheres[1].material.diffuse   = Colorf32{ 1.f, 0.6f, 0.3f, 1.f };
    //primitives.spheres[1].material.emittance = Colorf32{ 0.f, 0.f, 0.f, 1.f };
    //primitives.spheres[1].material.reflectance = 0.1f;

    primitives.planes[0].position = Math::Vec4f32{ 0.f, -.25f, 0.f, 0.f};
    primitives.planes[0].normal   = Math::Vec4f32{ 0.f, 1.f, 0.f, 0.f};
    primitives.planes[0].material.diffuse   = Math::Colorf32{1.f, 1.f, 1.f, 1.f};
    primitives.planes[0].material.emittance = Math::Colorf32{0.25f, 0.25f, 0.25f, 1.f};

    primitives.planes[1].position = Math::Vec4f32{ 0.f, 0.f, 1.f, 0.f};
    primitives.planes[1].normal   = Math::Vec4f32{ 0.f, 0.f, -1.f, 0.f};
    primitives.planes[1].material.diffuse   = Math::Colorf32{0.75f, 0.75f, 0.75f, 1.f};
    primitives.planes[1].material.emittance = Math::Colorf32{0.f, 0.f, 0.f, 1.f};
    primitives.planes[1].material.reflectance = 1.f;
    primitives.planes[1].material.roughness   = 0.f;

    primitives.planes[2] = primitives.planes[1];
    primitives.planes[2].position = Math::Vec4f32{1.2f, 0.f, 0.f, 0.f};
    primitives.planes[2].normal   = Math::Vec4f32{-1.f, 0.f, 0.f, 0.f};
    primitives.planes[2].material.diffuse = {1.f, 0.f, 0.f, 1.f};
    primitives.planes[2].material.roughness   = 1.0f;
    primitives.planes[2].material.reflectance = 0.0f;

    primitives.planes[3] = primitives.planes[1];
    primitives.planes[3].position = Math::Vec4f32{-1.2f, 0.f, 0.f, 0.f};
    primitives.planes[3].normal   = Math::Vec4f32{1.f, 0.f, 0.f, 0.f};
    primitives.planes[3].material.roughness   = 0.0f;
    primitives.planes[3].material.reflectance = 1.0f;

    primitives.planes[4].position = Math::Vec4f32{ 0.f, 0.f, renderParams.camera.position.z - 1.f, 0.f};
    primitives.planes[4].normal   = Math::Vec4f32{ 0.f, 0.f, 1.f, 0.f};
    primitives.planes[4].material.diffuse   = Math::Colorf32{.75f, .75f, .75f, 1.f};
    primitives.planes[4].material.emittance = Math::Colorf32{0.25f, 0.25f, 0.25f, 1.f};

    Material bunnyMaterial;
    bunnyMaterial.diffuse   = Math::Colorf32{.75f, .75f, .75f, 1.f};
    bunnyMaterial.emittance = Math::Colorf32{0.f, 0.f, 0.f, 1.f};
    bunnyMaterial.reflectance = 0.f;
    bunnyMaterial.roughness = 1.0f;
    bunnyMaterial.transparency = 0.0f;
    bunnyMaterial.index_of_refraction = 1.0f;

   //auto bunnyTriangles = LoadObjectFile("res/bunny.obj", bunnyMaterial);
   //for (auto tr : bunnyTriangles) {
   //    tr.v0 *= 10; tr.v0.y -= 0.5f; tr.v0.z *= -1.f; tr.v0.x *= -1.f; tr.v0.x -= 0.1f;
   //    tr.v1 *= 10; tr.v1.y -= 0.5f; tr.v1.z *= -1.f; tr.v1.x *= -1.f; tr.v1.x -= 0.1f;
   //    tr.v2 *= 10; tr.v2.y -= 0.5f; tr.v2.z *= -1.f; tr.v2.x *= -1.f; tr.v2.x -= 0.1f;
   //    primitives.triangles.Append(tr);
   //}

    PolarTracer pt(renderParams, primitives, {});

    const auto startTime = std::chrono::high_resolution_clock::now();
    pt.RayTraceScene(image);
    const auto endTime   = std::chrono::high_resolution_clock::now();

    const double duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.f;

    std::cout << std::fixed << '\n';
    std::cout << "Took " << duration << "s to render a " << WIDTH << " by " << HEIGHT << " image at " << SPP << " SPP with a maximum recursion depth of " << MAX_REC << ".\n";
    std::cout << "Big numbers:\n";
    std::cout << "-->" << (unsigned int)(WIDTH * HEIGHT) << " pixels.\n";
    std::cout << "-->" << (unsigned int)(WIDTH * HEIGHT * SPP) << " samples.\n";
    
    std::cout << "-->" << ((std::uint64_t)WIDTH * HEIGHT * SPP * MAX_REC) << " photons.\n";
    std::cout << "Timings:\n";
    std::cout << "-->" << (std::uint64_t)((WIDTH * HEIGHT * SPP) / duration) << " samples per sec.\n";
    std::cout << "-->" << (((std::uint64_t)WIDTH * HEIGHT * SPP * MAX_REC) / duration) << " photon paths per sec.\n";

    SaveImage(image, "frame");
}
