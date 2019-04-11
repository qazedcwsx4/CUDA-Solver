#include "Matrix.cuh"

Matrix::Matrix(int cols, int rows) : cols(cols), rows(rows)
{
	cudaMallocManaged(&mat, cols * rows);
}

Matrix* Matrix::ZeroCPU(int cols, int rows)
///zdaje mi sie, ze metoda bedzie wykonywania stosunkowo niewiele razy wiec nie potrzebuje zrownoleglenia.
{
	Matrix* ret = new Matrix(cols, rows);
	/*for (long i = 0; i < cols * rows; i++)
	{
		ret->mat[i] = 0.0f;
	}*/

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			ret->mat[i] = 0.0f;
		}
	}

	return ret;
}

__global__

void ZeroGPUKernel(int n, float* A)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		A[index] = 0.0f;
	}
}

Matrix* Matrix::ZeroGPU(int cols, int rows)
{
	Matrix* ret = new Matrix(cols, rows);
	int blockCount = (cols * rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
	ZeroGPUKernel <<<blockCount, BLOCK_SIZE >>>(cols * rows, ret->mat);
	cudaDeviceSynchronize();
	return ret;
}


Matrix* Matrix::separateDiagonal()
{
	if (cols != rows) throw "Matrix is not square";
	Matrix* ret = Matrix::ZeroCPU(cols, rows);
	for (int i = 0; i < cols; ++i)
	{
		ret->mat[i * cols + i] = this->mat[i * cols + i];
		this->mat[i * cols + i] = 0.0f;
	}
	return ret;
}

void Matrix::print() const
{
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			printf("%f ", mat[i * cols + j]);
		}
		printf("\n");
	}
}

Matrix::~Matrix()
{
	cudaFree(mat);
}
