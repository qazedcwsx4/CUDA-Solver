#include "Matrix.cuh"
#include <cstring>
#define Zero ZeroCPU

Matrix::Matrix(int cols, int rows) : cols(cols), rows(rows)
{
	printf("Matrix constructor\n");
	cudaMallocManaged(&mat, cols * rows * sizeof(float));
}

Matrix::Matrix(int cols, int rows, float* mat) : cols(cols), rows(rows), mat(mat)
{
	printf("Matrix constructor\n");
	cudaMallocManaged(&mat, cols * rows * sizeof(float));
}

Matrix::Matrix(const Matrix& a)
{
	printf("Matrix copy constructor\n");
	rows = a.rows;
	cols = a.cols;
	cudaMallocManaged(&mat, cols * rows * sizeof(float));
	std::memcpy(mat, a.mat, cols * rows * sizeof(float));
}

int Matrix::operator=(const Matrix& M)
{
	return 1;
}

Matrix* Matrix::ZeroCPU(int cols, int rows)
///zdaje mi sie, ze metoda bedzie wykonywania stosunkowo niewiele razy wiec nie potrzebuje zrownoleglenia.
{
	Matrix* ret = new Matrix(cols, rows);

	for (long i = 0; i < cols * rows; i++)
	{
		ret->mat[i] = 0.0f;
	}

	return ret;
}

Matrix* Matrix::OneCPU(int cols, int rows)
///zdaje mi sie, ze metoda bedzie wykonywania stosunkowo niewiele razy wiec nie potrzebuje zrownoleglenia.
{
	Matrix* ret = new Matrix(cols, rows);

	for (long i = 0; i < cols * rows; i++)
	{
		ret->mat[i] = 1.0f;
	}

	return ret;
}

__global__ void ZeroGPUKernel(const int n, float* A)
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

Matrix* Matrix::IdentityCPU(int cols, int rows)
{
	if (cols != rows) throw "Identity matrix must be square";
	Matrix* ret = Zero(cols, rows);
	for (int i = 0; i < cols; ++i)
	{
		ret->mat[i * cols + i] = 1.0f;
	}
	return ret;
}

Matrix* Matrix::separateDiagonal()
{
	if (cols != rows) throw "Matrix is not square";
	Matrix* ret = Zero(cols, rows);
	for (int i = 0; i < cols; ++i)
	{
		ret->mat[i * cols + i] = mat[i * cols + i];
		mat[i * cols + i] = 0.0f;
	}
	return ret;
}

void Matrix::inverseDiagonalInPlaceCPU()
{
	if (cols != rows) throw "Matrix is not square";
	for (int i = 0; i < cols; ++i)
	{
		if (mat[i * cols + i] == 0) throw "0 on diagonal";
		mat[i * cols + i] = 1 / mat[i * cols + i];
	}
}

Matrix Matrix::lu()
{
	throw "Not implemented";
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
	printf("\n");
}

Matrix::~Matrix()
{
	printf("Matrix destructor\n");
	cudaFree(mat);
}

__global__ void mulKernel(const int commonDim, const int cols, const int n, float* A, float* B, float* C)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int row = index / cols;
	int col = index % cols;

	for (int j = index; j < n; j += stride)
	{
		C[index] = 0;
		for (int i = 0; i < commonDim; i++)
		{
			C[index] += A[row * commonDim + i] * B[i * cols + col];
		}
	}
}

Matrix* operator*(const Matrix& a, const Matrix& b)
{
	if (a.cols != b.rows) throw "wrong dimensions for multiplication";
	auto ret = new Matrix(b.cols, a.rows);
	int blockCount = (a.rows * b.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
	printf("Matrix multiplication on %d blocks x %d threads\n", blockCount, BLOCK_SIZE);
	mulKernel <<< blockCount, BLOCK_SIZE >>>(a.cols, ret->cols, ret->cols * ret->rows, a.mat, b.mat, ret->mat);
	cudaDeviceSynchronize();
	return ret;
}

__global__ void addKernel(const int n, float* A, float* B, float* C)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		C[index] = A[index] + B[index];
	}
}

Matrix* operator+(const Matrix& a, const Matrix& b)
{
	if (a.cols != b.cols || a.rows != b.rows) throw "dimensions must equal for addition";
	auto ret = new Matrix(a.cols, a.rows);
	int blockCount = (a.rows * b.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
	printf("Matrix addition on %d blocks x %d threads\n", blockCount, BLOCK_SIZE);
	addKernel <<< blockCount, BLOCK_SIZE >>> (ret->cols * ret->rows, a.mat, b.mat, ret->mat);
	cudaDeviceSynchronize();
	return ret;
}

__global__ void subKernel(const int n, float* A, float* B, float* C)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		C[index] = A[index] - B[index];
	}
}

Matrix* operator-(const Matrix& a, const Matrix& b)
{
	if (a.cols != b.cols || a.rows != b.rows) throw "dimensions must equal for addition";
	auto ret = new Matrix(a.cols, a.rows);
	int blockCount = (a.rows * b.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
	printf("Matrix addition on %d blocks x %d threads\n", blockCount, BLOCK_SIZE);
	subKernel <<< blockCount, BLOCK_SIZE >>> (ret->cols * ret->rows, a.mat, b.mat, ret->mat);
	cudaDeviceSynchronize();
	return ret;
}


__global__ void additiveInverseKernel(const int n, float* A)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		C[index] = A[index] - B[index];
	}
}
Matrix* operator-(const Matrix& a)
{

}
