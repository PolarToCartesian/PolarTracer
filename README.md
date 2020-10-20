# PolarTracer (Cuda)

A ray tracer build with cuda and c++.

## Building Instructions

This project is built with clang and not nvcc in order to use C++17 features. This makes the building process non-trivial since clang doesn't support cuda versions newer than 9.0. On 64 bit linux systems, you have to:

+ Install CUDA 9.0 (not a newer version)
+ Install Clang++ (any version supporting C++17 should do)

On debian based systems, you can install the following useful tools: ```sudo apt install git imagemagick make clang```

Then, you can just run ```sh run.sh``` once you have cloned the repository with git.

## References

+ [NVIDIA's Fermi GPU Architecture White Paper](https://www.nvidia.fr/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf)
+ [CppCon 2016: “Bringing Clang and C++ to GPUs: An Open-Source, CUDA-Compatible GPU C++ Compiler"](https://www.youtube.com/watch?v=KHa-OSrZPGo)
