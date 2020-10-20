NVCC_FLAGS := -O3 --std=c++14
CLANG_CUDA_FLAGS := -L/opt/cuda/lib64 -L/usr/local/cuda/lib64 -lcuda -lcudart
CLANG_MISC_FLAGS := -std=c++17 -O3 -Wall -Wpedantic

all:
	clang++ -std=c++17 --cuda-gpu-arch=sm_35 -L/usr/local/cuda-9.0/lib64 ./src/Main.cu -o ./PolarTracer -lcudart_static -ldl -lrt -pthread

clean:
	rm -f ./build/*