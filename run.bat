nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64" --std=c++17 src/PolarTracer.cu -o PolarTracer
PolarTracer.exe
magick convert frame.pam frame.png
del frame.pam
frame.png