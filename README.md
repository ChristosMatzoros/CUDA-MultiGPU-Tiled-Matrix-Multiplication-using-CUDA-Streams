# CUDA-MultiGPU-Tiled-Matrix-Multiplication-using-CUDA-Streams
CUDA application that uses multiple GPUs to compute matrix-matrix multiplication. The matrix is tiled to run from 1 to 8 devices. This code was part of my Bachelor thesis: "A Study on the Computational Exploitation of Remote Virtualized Graphics Cards" (https://bit.ly/37tIG0D)


To run the program properly alter the CUDA_PATH variable accordingly on the Makefile.

Compile:
make

Run:
./matrixMulMultiGPU
