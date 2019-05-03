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
	//static Matrix ZeroGPU(int cols, int rows);
	static Matrix IdentityCPU(int cols, int rows);
	static Matrix FromFile(std::string path);
	static Matrix Jacobi(const Matrix& A, const Matrix& b);
	static Matrix ForwardSubstitution(const Matrix& A, const Matrix& b);
	static Matrix BackwardSubstitution(const Matrix& A, const Matrix& b);
	static Matrix GaussSeidel(const Matrix& A, const Matrix& b);
	static Matrix LUMehtod(const Matrix& A, const Matrix& b);

	//nowy pomysl
	static void doolitle(Matrix& L, Matrix& U, const Matrix& A);

	void toFile(std::string path);
	friend Matrix operator*(const Matrix& a, const Matrix& b);
	friend Matrix operator+(const Matrix& a, const Matrix& b);
	friend Matrix operator-(const Matrix& a, const Matrix& b);
	friend Matrix operator-(const Matrix& a);
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
};
