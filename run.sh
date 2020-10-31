echo "Compiling..."
nvcc -O3 -ccbin "/usr/bin/clang++" -std=c++17 src/PolarTracer.cu -o PolarTracer
echo "Running..."
./PolarTracer
echo "Converting & Deleting..."
convert frame.pam frame.png
rm frame.pam
eog frame.png > /dev/null 2>&1
