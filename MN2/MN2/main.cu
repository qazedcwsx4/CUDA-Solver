#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#include <stdio.h>
#include "Matrix.cuh"
#include <vector>


int main()
{
	/*auto A = Matrix::OneCPU(10, 10);
	auto LU = A;


	auto x = Matrix::ZeroCPU(1, 10);

	float matB[] = {1, 2, 3, 4};
	auto b = Matrix::ZeroCPU(1, 10);
	auto I = Matrix::IdentityCPU(10, 10);
	auto invD = A->separateDiagonal();
	invD->inverseDiagonalInPlaceCPU();
	invD->print();


	//stworzylem potwora
	//x = *(*(*-*invD * *LU) * *x) + *(*-*invD * *b);*/


	auto A = Matrix::FromFile("A2.txt");
	//auto x = Matrix::ZeroCPU(1, 4);
	auto b = Matrix::FromFile("b2.txt");
	auto x = Matrix::Stub();
	//auto DT = Matrix::FromFile("doolitletest.txt");
	



	auto start = std::chrono::steady_clock::now();
	x = Matrix::Jacobi(A, b);
	auto end = std::chrono::steady_clock::now();
	printf("Jacobi method: %lld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
	x.print();

	start = std::chrono::steady_clock::now();
	x = Matrix::GaussSeidel(A, b);
	end = std::chrono::steady_clock::now();
	printf("Gauss-Seidel method: %lld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
	x.print();

	start = std::chrono::steady_clock::now();
	x = Matrix::LUMehtod(A, b);
	end = std::chrono::steady_clock::now();
	printf("LU factorization method: %lld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
	x.print();

	//system("pause");
	return 0;
}
