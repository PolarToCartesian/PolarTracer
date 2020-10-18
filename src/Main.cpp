#include "Common.hpp"

// From CudaRTX.cu
extern void RayTraceScene(Image& cpuOutputImage);

int main(int argc, char** argv) {
  Image image(1920, 1080);
  RayTraceScene(image);
  image.Save("frame");
}
