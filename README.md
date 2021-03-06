# PolarTracer (Cuda)

![Sample Image](samples/sample.png)
![Sample Image](samples/sample2.png)

A Path Tracer built with Cuda and C++. For more information about path tracers, you can visit a previous project of mine [CPU-Path-Tracer](https://github.com/PolarToCartesian/CPU-Path-Tracer)

## Building Instructions

+ Download & Install the [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads).
+ Install your C++ compiler of choice (compatible with nvcc, NVIDIA's CUDA compiler).
+ Install imagemagick (On Windows & Linux): this is used to convert *.pam images into *.png files

+ On windows, run ```run.bat``` to build & run the test scene.
+ On linux, run ```run.sh``` to build & run the test scene.

+ The frame will be saved as ```frame.png```.

+ Enjoy :-)

## References

+ [NVIDIA's Fermi GPU Architecture White Paper](https://www.nvidia.fr/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf)
+ [CppCon 2016: “Bringing Clang and C++ to GPUs: An Open-Source, CUDA-Compatible GPU C++ Compiler"](https://www.youtube.com/watch?v=KHa-OSrZPGo)
