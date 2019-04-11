#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#include <stdio.h>
#include "Matrix.cuh"

int main()
{
	auto start = std::chrono::steady_clock::now();
	Matrix *matrix = Matrix::ZeroCPU(1000, 1000);
	auto end = std::chrono::steady_clock::now();
	printf("Elapsed time: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
	//matrix->print();
}
