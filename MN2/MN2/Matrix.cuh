#pragma once

#define BLOCK_SIZE 256

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>

class Matrix
{
private:
	unsigned int rows;
	unsigned int cols;
	double* mat;
public:
	//getters
	unsigned int getRows() const;
	unsigned int getCols() const;

	//constructors
	Matrix(int cols, int rows, double* mat);
	Matrix(int cols, int rows);
	Matrix(const Matrix& a);
	void operator=(const Matrix& a);
	static Matrix Stub();
	static Matrix ZeroCPU(int cols, int rows);
	static Matrix OneCPU(int cols, int rows);
	static Matrix ZeroGPU(int cols, int rows);
	static Matrix IdentityCPU(int cols, int rows);
	static Matrix FromFile(std::string path);
	static Matrix Jacobi(const Matrix& A, const Matrix& b);
	static Matrix JacobiOptimal(const Matrix& A, const Matrix& b);

	static Matrix ForwardSubstitution(const Matrix& A, const Matrix& b);
	static Matrix BackwardSubstitution(const Matrix& A, const Matrix& b);
	static Matrix GaussSeidel(const Matrix& A, const Matrix& b);
	static Matrix GaussSeidelOptimal(const Matrix& A, const Matrix& b);
	static Matrix LUMehtod(const Matrix& A, const Matrix& b);
	static Matrix LUMehtodOptimal(const Matrix& A, const Matrix& b);


	//nowy pomysl
	static void doolitle(Matrix& L, Matrix& U, const Matrix& A);
	static void doolitleGPU(Matrix& L, Matrix& U, const Matrix& A);
	static void createTest(Matrix& A, Matrix& b, Matrix& x, int size);
	static void createTask(Matrix& A, Matrix& b, const int size);
	static void createTaskC(Matrix& A, Matrix& b);

	static double maxError(Matrix& x, Matrix& r);
	static void copyGPU(Matrix& a, const Matrix& b);
	static void separateDiagonalAndInverseGPU(Matrix& d, Matrix& A);
	static void separateUpperGPU(Matrix& U, Matrix& A);
	static void additiveInverseInPlaceGPU(Matrix& A);
	static void forwardSubstitutionGPU(Matrix& result, const Matrix& A, const Matrix& b);
	static void backwardSubstitutionGPU(Matrix& result, const Matrix& A, const Matrix& b);


	void toFile(std::string path);
	Matrix separateDiagonal();
	Matrix diagonalCPU() const;
	Matrix lowerCPU() const;
	Matrix upperCPU() const;
	void inverseDiagonalInPlaceCPU();
	void transposeVectorInPlace();
	double vectorEuclideanNorm();
	Matrix lu();
	void print() const;
	~Matrix();

	friend Matrix operator*(const Matrix& a, const Matrix& b);
	friend Matrix operator+(const Matrix& a, const Matrix& b);
	friend Matrix operator-(const Matrix& a, const Matrix& b);
	friend Matrix operator-(const Matrix& a);
	static void refMul(Matrix& result, const Matrix& a, const Matrix& b);
	static void refMulDiag(Matrix& result, const Matrix& a, const Matrix& b);
	static void refAdd(Matrix& result, const Matrix& a, const Matrix& b);
	static void refSub(Matrix& result, const Matrix& a, const Matrix& b);
};
