#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#include <stdio.h>
#include "Matrix.cuh"


int main()
{
	auto start = std::chrono::steady_clock::now();

	//auto A = new Matrix(10, 10);
	auto A = Matrix::OneCPU(10, 10);
	auto LU = A;


	auto x = Matrix::ZeroCPU(1, 10);

	float matB[] = {1, 2, 3, 4};
	auto b = Matrix::ZeroCPU(1, 10);
	auto I = Matrix::IdentityCPU(10, 10);
	auto invD = A->separateDiagonal();
	invD->inverseDiagonalInPlaceCPU();
	invD->print();


	//stworzylem potwora
	//x = *(*(*-*invD * *LU) * *x) + *(*-*invD * *b);

	auto end = std::chrono::steady_clock::now();
	printf("Elapsed time: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
	return 0;
}
