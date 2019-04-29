#pragma once

#define BLOCK_SIZE 256

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

class Matrix
{
private:
	unsigned int rows;
	unsigned int cols;
	float* mat;
public:
	Matrix(int cols, int rows, float* mat);
	Matrix(int cols, int rows);
	Matrix(const Matrix& a);
	int operator=(const Matrix &M);
	static Matrix *ZeroCPU(int cols, int rows);
	static Matrix *OneCPU(int cols, int rows);
	static Matrix *ZeroGPU(int cols, int rows);
	static Matrix *IdentityCPU(int cols, int rows);
	friend Matrix *operator*(const Matrix& a, const Matrix& b);
	friend Matrix *operator+(const Matrix& a, const Matrix& b);
	friend Matrix *operator-(const Matrix& a, const Matrix& b);
	friend Matrix *operator-(const Matrix& a);


	Matrix *separateDiagonal();
	void inverseDiagonalInPlaceCPU();
	Matrix lu();
	void print() const;
	~Matrix();
};
