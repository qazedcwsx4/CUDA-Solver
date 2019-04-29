#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#include <stdio.h>
#include "Matrix.cuh"


int main()
{
	auto start = std::chrono::steady_clock::now();
	Matrix *a = Matrix::OneCPU(2, 2);
	Matrix *b = Matrix::OneCPU(2, 2);
	auto c = *a*b;
	auto end = std::chrono::steady_clock::now();
	printf("Elapsed time: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
	matrix->separateDiagonal();
	matrix->print();
	return 0;
}
