NVCC_FLAGS := -O3 --std=c++14
CLANG_CUDA_FLAGS := -L/opt/cuda/lib64 -L/usr/local/cuda/lib64 -lcuda -lcudart
CLANG_MISC_FLAGS := -std=c++17 -O3 -Wall -Wpedantic

all:
	nvcc ${NVCC_FLAGS} ./src/CudaRTX.cu -o ./PolarTracer

clean:
	rm -f ./build/*