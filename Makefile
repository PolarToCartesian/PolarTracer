NVCC_FLAGS := 
CLANG_CUDA_FLAGS := -L/opt/cuda/lib64 -L/usr/local/cuda/lib64 -lcuda -lcudart
CLANG_MISC_FLAGS := -O3 -Wall -Wpedantic

all:
	nvcc --std=c++14 ./src/GPUPT.cu -O3 -c -o ./build/GPUPT.o
	clang++ ${CLANG_CUDA_FLAGS} ${CLANG_MISC_FLAGS} -c ./src/Launcher.cpp -o ./build/Launcher.o
	clang++ ${CLANG_CUDA_FLAGS} ${CLANG_MISC_FLAGS} -o ./bin/PolarTracer ./build/*.o

clean:
	rm -f ./build/*