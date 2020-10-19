make
./PolarTracer
convert frame.pam frame.png
rm frame.pam
eog frame.png > /dev/null 2>&1