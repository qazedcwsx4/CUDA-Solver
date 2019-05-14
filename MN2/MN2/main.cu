#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#include <stdio.h>
#include "Matrix.cuh"
#include <vector>


int main()
{
	Matrix A = Matrix::Stub();
	Matrix b = Matrix::Stub();
	Matrix x = Matrix::Stub();

	//Matrix::createTask(A, b, 994);
	//Matrix::createTask(A, b, 3000);
	Matrix::createTest(A, b, x, 1000);


	//wywolanie przed zeby przygotowac device
	Matrix::JacobiOptimal(A, b);
	auto start = std::chrono::steady_clock::now();
	x = Matrix::JacobiOptimal(A, b);
	auto end = std::chrono::steady_clock::now();
	printf("Jacobi method: %lld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

	/*Matrix::GaussSeidelOptimal(A, b);
	start = std::chrono::steady_clock::now();
	x = Matrix::GaussSeidelOptimal(A, b);
	end = std::chrono::steady_clock::now();
	printf("Gauss-Seidel method: %lld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

	Matrix::LUMehtodOptimal(A, b);
	start = std::chrono::steady_clock::now();
	x = Matrix::LUMehtodOptimal(A, b);
	end = std::chrono::steady_clock::now();
	printf("LU method: %lld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
	printf("residue: %f\n",(A * x - b).vectorEuclideanNorm());*/



	//system("pause");
	return 0;
}
