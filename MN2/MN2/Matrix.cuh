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
	Matrix(int cols, int rows);
	static Matrix* ZeroCPU(int cols, int rows);
	static Matrix* OneCPU(int cols, int rows);
	static Matrix* ZeroGPU(int cols, int rows);
	friend Matrix* operator*(const Matrix& a, const Matrix* b);
	Matrix* separateDiagonal();
	Matrix* lu();
	void print() const;
	~Matrix();
};
