nvcc -O3 -Xcudafe "--diag_suppress=implicit_return_from_non_void_function" -Xcompiler "" --std=c++17 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29333\bin\Hostx64\x64" src/EntryPoint.cu -o PolarTracer.exe
PolarTracer.exe
magick convert frame.pam frame.png
del frame.pam