all:
	nvcc --std=c++14 ./PolarTracer.cu -O3 -o ./PolarTracer

clean:
	rm -f ./build/*