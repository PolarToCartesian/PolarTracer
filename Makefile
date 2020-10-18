NVCC_FLAGS := -O3 -c --std=c++14
CLANG_CUDA_FLAGS := -L/opt/cuda/lib64 -L/usr/local/cuda/lib64 -lcuda -lcudart
CLANG_MISC_FLAGS := -std=c++17 -O3 -Wall -Wpedantic

all:
	nvcc ${NVCC_FLAGS} ./src/CudaRTX.cu -o ./build/CudaRTX.o
	clang++ ${CLANG_CUDA_FLAGS} ${CLANG_MISC_FLAGS} -o ./PolarTracer ./src/Main.cpp ./build/CudaRTX.o

clean:
	rm -f ./build/*