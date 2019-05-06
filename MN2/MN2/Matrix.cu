#include "Matrix.cuh"
#include <cstring>
#include <fstream>
#include <ctime>
#include <device_functions.h>

#ifdef __CUDACC__
#define cuda_SYNCTHREADS() __syncthreads()
#else
#define cuda_SYNCTHREADS()
#endif

#define Zero ZeroCPU
#define PRINT_LOG false
//#define TARGET_RESIDUE ((double)1.0e-9);

const double TARGET_RESIDUE = 1.0e-6;

Matrix::Matrix(int cols, int rows) : cols(cols), rows(rows)
{
	if (PRINT_LOG) printf("Matrix constructor\n");
	cudaMallocManaged(&mat, cols * rows * sizeof(double));
}

unsigned Matrix::getRows() const
{
	return rows;
}

unsigned Matrix::getCols() const
{
	return cols;
}

Matrix::Matrix(int cols, int rows, double* mat) : cols(cols), rows(rows), mat(mat)
{
	if (PRINT_LOG) printf("Matrix constructor\n");
	//cudaMallocManaged(&mat, cols * rows * sizeof(double));
}

Matrix::Matrix(const Matrix& a)
{
	if (PRINT_LOG) printf("Matrix copy constructor\n");
	rows = a.rows;
	cols = a.cols;
	cudaMallocManaged(&mat, cols * rows * sizeof(double));
	std::memcpy(mat, a.mat, cols * rows * sizeof(double));
}

void Matrix::operator=(const Matrix& a)
{
	if (PRINT_LOG) printf("Matrix assignment operator\n");
	rows = a.rows;
	cols = a.cols;
	cudaFree(mat);
	cudaMallocManaged(&mat, cols * rows * sizeof(double));
	std::memcpy(mat, a.mat, cols * rows * sizeof(double));
}

Matrix Matrix::Stub()
{
	return Matrix(1, 1);
}

Matrix Matrix::ZeroCPU(int cols, int rows)
{
	double* mat;
	cudaMallocManaged(&mat, cols * rows * sizeof(double));
	cudaDeviceSynchronize();
	for (long i = 0; i < cols * rows; i++)
	{
		mat[i] = 0.0f;
	}
	return Matrix(cols, rows, mat);
}

Matrix Matrix::OneCPU(int cols, int rows)
{
	double* mat;
	cudaMallocManaged(&mat, cols * rows * sizeof(double));
	for (long i = 0; i < cols * rows; i++)
	{
		mat[i] = 1.0f;
	}
	return Matrix(cols, rows, mat);
}

__global__ void ZeroGPUKernel(const int n, double* A)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		A[index] = 0.0f;
	}
}

Matrix Matrix::ZeroGPU(int cols, int rows)
{
	double* mat;
	cudaMallocManaged(&mat, cols * rows * sizeof(double));
	int blockCount = (cols * rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
	ZeroGPUKernel <<<blockCount, BLOCK_SIZE >>>(cols * rows, mat);
	cudaDeviceSynchronize();
	return Matrix(cols, rows, mat);
}

Matrix Matrix::IdentityCPU(int cols, int rows)
{
	if (cols != rows) throw "Identity matrix must be square";
	auto ret = Zero(cols, rows);
	for (int i = 0; i < cols; ++i)
	{
		ret.mat[i * cols + i] = 1.0f;
	}
	return ret;
}

Matrix Matrix::FromFile(std::string path)
{
	std::fstream reader;
	int cols, rows;
	reader.open(path, std::ios::in);
	reader.seekp(0);
	reader >> cols;
	reader >> rows;
	double* mat;
	cudaMallocManaged(&mat, cols * rows * sizeof(double));
	for (int i = 0; i < cols * rows; ++i)
	{
		reader >> mat[i];
	}
	reader.close();
	return Matrix(cols, rows, mat);
}

Matrix Matrix::Jacobi(const Matrix& A, const Matrix& b)
{
	auto LU = A;
	auto invD = (LU.separateDiagonal());
	auto x = ZeroCPU(1, A.getRows());
	invD.inverseDiagonalInPlaceCPU();
	auto M = -invD * LU;
	auto temp = invD * b;
	double res;
	int counter = 9;

	do
	{
		x = (M * x + temp);
		if (counter++ == 9)
		{
			counter = 0;
			res = (A * x - b).vectorEuclideanNorm();
			printf("res: %f\n", res);
		}
	}
	while (res > TARGET_RESIDUE);
	return x;
}

Matrix Matrix::JacobiOptimal(const Matrix& A, const Matrix& b)
{
	// 25% czasu wykonania (80000us) prawdopodobnie kopiowanie pamieci z device na host i z powrotem
	//auto LU = A;
	//->
	auto LU = Matrix(A.cols, A.rows);
	copyGPU(LU, A);
	//32x wzrost wydajnosci

	//auto invD = (LU.separateDiagonal());
	//invD.inverseDiagonalInPlaceCPU();
	auto invD = Matrix(A.cols, A.rows);
	separateDiagonalAndInverseGPU(invD, LU);

	auto x = ZeroGPU(1, A.getRows());

	//auto temp1 = invD * b;
	auto temp1 = Matrix(1, A.rows);
	refMul(temp1, invD, b);

	//auto M = -invD * LU;
	//auto M = Matrix(A.cols, A.rows);
	auto M = Matrix(A.cols, A.rows);
	additiveInverseInPlaceGPU(invD);
	refMulDiag(M, invD, LU);


	double res = 100;
	int counter = 9;

	auto memmul = Matrix(1, A.rows);

	auto memmulres = Matrix(1, A.rows);
	auto resVector = Matrix(1, A.rows);

	do
	{
		refMul(memmul, M, x);
		refAdd(x, memmul, temp1);
		//x = (M * x + temp);
		if (counter++ == 9)
		{
			counter = 0;
			refMul(memmulres, A, x);
			refSub(resVector, memmulres, b);
			res = resVector.vectorEuclideanNorm();
			//printf("res: %f\n", res);
		}
	}
	while (res > TARGET_RESIDUE);
	return x;
}

Matrix Matrix::ForwardSubstitution(const Matrix& A, const Matrix& b)
{
	if (!(A.cols == A.rows && A.rows == b.rows)) throw "Incorrect dimensions";
	auto x = Matrix(1, A.getRows());

	for (int i = 0; i < x.rows; ++i)
	{
		double sum = 0;
		for (int j = 0; j < i; ++j)
		{
			sum += A.mat[i * A.cols + j] * x.mat[j];
		}
		x.mat[i] = (b.mat[i] - sum) / A.mat[i * A.cols + i];
	}
	return x;
}

Matrix Matrix::BackwardSubstitution(const Matrix& A, const Matrix& b)
{
	if (!(A.cols == A.rows && A.rows == b.rows)) throw "Incorrect dimensions";
	auto x = Matrix(1, A.getRows());

	x.mat[0] = b.mat[0] / A.mat[0];

	for (int i = x.rows - 1; i >= 0; --i)
	{
		double sum = 0;
		for (int j = i + 1; j < A.cols; ++j)
		{
			sum += A.mat[i * A.cols + j] * x.mat[j];
		}
		x.mat[i] = (b.mat[i] - sum) / A.mat[i * A.cols + i];
	}
	return x;
}

Matrix Matrix::GaussSeidel(const Matrix& A, const Matrix& b)
{
	auto DL = -(A.lowerCPU() + A.diagonalCPU());
	auto U = A.upperCPU();
	auto x = ZeroCPU(1, A.getRows());
	auto temp = Matrix::ForwardSubstitution(DL, b);
	double res;
	int counter = 9;

	do
	{
		//x = -(Matrix::ForwardSubstitution(DL, U * x)) + temp;
		x = (Matrix::ForwardSubstitution(DL, U * x)) + temp;
		//if (counter++ == 9)
		//{
		//	counter = 0;
		res = (A * (-x) - b).vectorEuclideanNorm();
		//}
		//printf("res: %f \n", res);
		//(x).print();
	}
	while (res > TARGET_RESIDUE);
	return -x;
}

Matrix Matrix::GaussSeidelOptimal(const Matrix& A, const Matrix& b)
{
	//auto DL = (A.lowerCPU() + A.diagonalCPU());
	//auto U = A.upperCPU();
	/*auto DL = Matrix(A.cols, A.rows);
	auto U = Matrix(A.cols, A.rows);
	copyGPU(DL, A);
	separateUpperGPU(U, DL);*/

	auto DL = (A.lowerCPU() + A.diagonalCPU());
	auto U = A.upperCPU();

	auto x = ZeroCPU(1, A.getRows());
	auto temp = Matrix::ForwardSubstitution(DL, b);

	auto memmul = Matrix(1, A.rows);
	auto memforwardsub = Matrix(1, A.rows);


	auto memmulres = Matrix(1, A.rows);
	auto resVector = Matrix(1, A.rows);
	double res;
	int counter = 9;

	do
	{
		//x = -(Matrix::ForwardSubstitution(DL, U * x)) + temp;
		refMul(memmul, U, x);

		forwardSubstitutionGPU(memforwardsub, DL, memmul);
		//memforwardsub = Matrix::ForwardSubstitution(DL, memmul);

		//double xd = maxError(memforwardsub, memforwardsub2);

		additiveInverseInPlaceGPU(memforwardsub);
		refAdd(x, memforwardsub, temp);

		//x = memforwardsub + temp;
		if (counter++ == 9)
		{
			counter = 0;
			refMul(memmulres, A, x);
			refSub(resVector, memmulres, b);
			res = resVector.vectorEuclideanNorm();
		}
		printf("res: %f \n", res);
		//(x).print();
	}
	while (res > TARGET_RESIDUE);
	return x;
}

Matrix Matrix::LUMehtod(const Matrix& A, const Matrix& b)
{
	Matrix L = Matrix::Stub();
	Matrix U = Matrix::Stub();

	Matrix::doolitle(L, U, A);

	auto y = Matrix::ForwardSubstitution(L, b);

	return Matrix::BackwardSubstitution(U, y);
}

Matrix Matrix::LUMehtodOptimal(const Matrix& A, const Matrix& b)
{
	Matrix L = Matrix::Stub();
	Matrix U = Matrix::Stub();

	Matrix::doolitle(L, U, A);

	auto y = Matrix::ForwardSubstitution(L, b);

	return Matrix::BackwardSubstitution(U, y);
}

void Matrix::doolitle(Matrix& L, Matrix& U, const Matrix& A)
{
	if (A.cols != A.rows) throw "Matrix is not square";
	L = OneCPU(A.cols, A.rows).diagonalCPU();
	U = ZeroCPU(A.cols, A.rows);
	for (int j = 0; j < A.cols; ++j)
	{
		for (int i = 0; i <= j; ++i)
		{
			double sum = 0;
			for (int k = 0; k < i; ++k)
			{
				sum += L.mat[i * L.cols + k] * U.mat[k * U.cols + j];
			}
			U.mat[i * U.cols + j] = A.mat[i * U.cols + j] - sum;
		}

		for (int i = j + 1; i < A.cols; ++i)
		{
			double sum = 0;
			for (int k = 0; k < j; ++k)
			{
				sum += L.mat[i * L.cols + k] * U.mat[k * U.cols + j];
			}
			L.mat[i * U.cols + j] = 1 / U.mat[j * U.cols + j] * (A.mat[i * U.cols + j] - sum);
		}
	}
}

__global__ void doolitleKernel(const int n, double* A, double* B)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		A[j] = B[j];
	}
}


void Matrix::doolitleGPU(Matrix& L, Matrix& U, const Matrix& A)
{
	int blockCount = (A.rows * A.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
	//doolitleKernel <<< blockCount, BLOCK_SIZE >>> (A.rows * A.cols, A.mat);
	cudaDeviceSynchronize();
}

void Matrix::createTest(Matrix& A, Matrix& b, Matrix& x, int size)
{
	srand(time(NULL));

	const int constrange = 100;
	const auto r = [](int range)-> double { return (double)(rand() % 20000) / 100 - 100; };
	x = Matrix(1, size);
	A = Matrix(size, size);
	b = Matrix(1, size);
	for (int i = 0; i < size; ++i)
	{
		x.mat[i] = r(100);
	}

	for (int i = 0; i < size; ++i)
	{
		double sum = 0;
		for (int j = 0; j < size; ++j)
		{
			if (i != j)
			{
				A.mat[i * size + j] = r(100);
				sum += fabs(A.mat[i * size + j]);
			}
			double randomized = r(100);
			if (randomized > 0)
			{
				A.mat[i * size + i] = sum + r(10);
			}
			else
			{
				A.mat[i * size + i] = -sum + r(10);
			}
		}
	}

	for (int i = 0; i < size; ++i)
	{
		double sum = 0;
		for (int j = 0; j < size; ++j)
		{
			sum += A.mat[i * size + j] * x.mat[j];
		}
		b.mat[i] = sum;
	}
}

void Matrix::createTask(Matrix& A, Matrix& b, const int size)
{
	//const int size = 994;
	const int a1 = 5 + 7;
	const int a2 = -1;
	const int a3 = a2;
	const int inSin(1 + 1);
	A = Matrix::ZeroCPU(size, size);
	b = Matrix(1, size);

	for (int i = 0; i < size; ++i)
	{
		A.mat[size * i + i] = a1;
		if (size * i + i - 1 >= 0)
			A.mat[size * i + i - 1] = a2;
		if (size * i + i - 2 >= 0)
			A.mat[size * i + i - 2] = a3;
		if (size * i + i + 1 < size * size)
			A.mat[size * i + i + 1] = a2;
		if (size * i + i + 2 < size * size)
			A.mat[size * i + i + 2] = a3;
	}

	for (int i = 0; i < size; ++i)
	{
		b.mat[i] = sin(i * inSin);
	}
}

void Matrix::createTaskC(Matrix& A, Matrix& b)
{
	const int size = 994;
	const int a1 = 3;
	const int a2 = -1;
	const int a3 = a2;
	const int inSin(1 + 1);
	A = Matrix::ZeroCPU(size, size);
	b = Matrix(1, size);

	for (int i = 0; i < size; ++i)
	{
		A.mat[size * i + i] = a1;
		if (size * i + i - 1 >= 0)
			A.mat[size * i + i - 1] = a2;
		if (size * i + i - 2 >= 0)
			A.mat[size * i + i - 2] = a3;
		if (size * i + i + 1 < size * size)
			A.mat[size * i + i + 1] = a2;
		if (size * i + i + 2 < size * size)
			A.mat[size * i + i + 2] = a3;
	}

	for (int i = 0; i < size; ++i)
	{
		b.mat[i] = sin(i * inSin);
	}
}

double Matrix::maxError(Matrix& x, Matrix& r)
{
	if (x.rows * x.cols != r.rows * r.cols) throw "Matrices are not the same size";
	double max = 0;
	for (int i = 0; i < x.rows * x.cols; ++i)
	{
		if (fabs(x.mat[i] - r.mat[i]) > max)
			max = fabs(x.mat[i] - r.mat[i]);
	}
	return max;
}

__global__ void copyKernel(const int n, double* A, double* B)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		A[j] = B[j];
	}
}

void Matrix::copyGPU(Matrix& a, const Matrix& b)
{
	int blockCount = (a.cols * a.rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
	copyKernel <<< blockCount, BLOCK_SIZE >>>(a.cols * a.rows, a.mat, b.mat);
	cudaDeviceSynchronize();
}

__global__ void separateDiagonalKernel(const int n, double* d, double* A)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		d[j * n + j] = 1 / A[j * n + j];
		A[j * n + j] = 0;
	}
}

void Matrix::separateDiagonalAndInverseGPU(Matrix& d, Matrix& A)
{
	int blockCount = (A.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
	separateDiagonalKernel <<< blockCount, BLOCK_SIZE >>>(A.cols, d.mat, A.mat);
	cudaDeviceSynchronize();
}

__global__ void separateUpperKernel(const int n, const int cols, double* U, double* A)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		int row = j / cols;
		int col = j % cols;
		if (col > row)
		{
			U[j] = A[j];
			A[j] = 0;
		}
	}
}

void Matrix::separateUpperGPU(Matrix& U, Matrix& A)
{
	int blockCount = (A.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
	separateUpperKernel <<< blockCount, BLOCK_SIZE >>>(A.cols * A.rows, A.cols, U.mat, A.mat);
	cudaDeviceSynchronize();
}

__global__ void additiveInverseInPlaceKernel(const int n, double* A)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		A[j] = -A[j];
	}
}

void Matrix::additiveInverseInPlaceGPU(Matrix& A)
{
	int blockCount = (A.rows * A.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
	additiveInverseInPlaceKernel <<< blockCount, BLOCK_SIZE >>>(A.rows * A.cols, A.mat);
	cudaDeviceSynchronize();
}

__global__ void forwardSubstitutionKernel(const int n, double* A, double* b, double* x)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		double sum = 0;
		for (int i = 0; i < n; i++)
		{
			if (i == j)
			{
				x[j] = (b[j] - sum) / A[j * n + j];
			}
			cuda_SYNCTHREADS();
			if (i < j)
			{
				sum += A[j * n + i] * x[i];
			}
		}
	}
}

void Matrix::forwardSubstitutionGPU(Matrix& result, const Matrix& A, const Matrix& b)
{
	int blockCount = 1;
	int blockSize = pow(2, ceil(log2f(A.cols)));
	forwardSubstitutionKernel <<< blockCount, blockSize >>>(A.cols, A.mat, b.mat, result.mat);
	cudaDeviceSynchronize();
}

void Matrix::backwardSubstitutionGPU(Matrix& result, const Matrix& A, const Matrix& b)
{
}

void Matrix::toFile(std::string path)
{
	std::fstream writer;
	writer.open(path, std::ios::out);
	writer.seekg(0);
	writer << cols << ' ' << rows << '\n';
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			writer << mat[i * cols + j] << ' ';
		}
		writer << "\n";
	}
	writer.close();
}

Matrix Matrix::separateDiagonal()
{
	if (cols != rows) throw "Matrix is not square";
	auto ret = Zero(cols, rows);
	for (int i = 0; i < cols; ++i)
	{
		ret.mat[i * cols + i] = mat[i * cols + i];
		mat[i * cols + i] = 0.0f;
	}
	return ret;
}

Matrix Matrix::diagonalCPU() const
{
	if (cols != rows) throw "Matrix is not square";
	auto ret = Zero(cols, rows);
	for (int i = 0; i < cols; ++i)
	{
		ret.mat[i * cols + i] = mat[i * cols + i];
	}
	return ret;
}

Matrix Matrix::lowerCPU() const
{
	if (cols != rows) throw "Matrix is not square";
	auto ret = Zero(cols, rows);
	for (int j = 0; j < cols; ++j)
	{
		for (int i = 0; i < j; ++i)
		{
			ret.mat[j * cols + i] = mat[j * cols + i];
		}
	}
	return ret;
}

Matrix Matrix::upperCPU() const
{
	if (cols != rows) throw "Matrix is not square";
	auto ret = Zero(cols, rows);
	for (int j = 0; j < cols; ++j)
	{
		for (int i = j + 1; i < cols; ++i)
		{
			ret.mat[j * cols + i] = mat[j * cols + i];
		}
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

void Matrix::transposeVectorInPlace()
{
	unsigned int tmp = cols;
	cols = rows;
	rows = tmp;
}

double Matrix::vectorEuclideanNorm()
{
	if (cols != 1 && rows != 1) throw "Matrix is not a vector";
	double sum = 0;
	for (int i = 0; i < cols * rows; ++i)
	{
		sum += mat[i] * mat[i];
	}
	return sqrt(sum);
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
	if (PRINT_LOG) printf("Matrix destructor\n");
	cudaFree(mat);
	//free(mat);
}

__global__ void mulKernel(const int commonDim, const int cols, const int n, double* A, double* B, double* C)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		int row = j / cols;
		int col = j % cols;
		C[j] = 0;
		for (int i = 0; i < commonDim; i++)
		{
			C[j] += A[row * commonDim + i] * B[i * cols + col];
		}
	}
}

void Matrix::refMul(Matrix& result, const Matrix& a, const Matrix& b)
{
	int blockCount = (a.rows * b.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
	mulKernel <<< blockCount, BLOCK_SIZE >>>(a.cols, b.cols, b.cols * a.rows, a.mat, b.mat, result.mat);
	cudaDeviceSynchronize();
}

__global__ void mulDiagKernel(const int commonDim, const int cols, const int n, double* A, double* B, double* C)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		int row = j / cols;
		int col = j % cols;
		C[j] = A[row * commonDim + row] * B[row * commonDim + col];
	}
}

void Matrix::refMulDiag(Matrix& result, const Matrix& a, const Matrix& b)
{
	int blockCount = (a.rows * b.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
	mulDiagKernel << < blockCount, BLOCK_SIZE >> >(a.cols, b.cols, b.cols * a.rows, a.mat, b.mat, result.mat);
	cudaDeviceSynchronize();
}

Matrix operator*(const Matrix& a, const Matrix& b)
{
	if (a.cols != b.rows) throw "wrong dimensions for multiplication";
	double* mat;
	cudaMallocManaged(&mat, b.cols * a.rows * sizeof(double));
	int blockCount = (a.rows * b.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
	if (PRINT_LOG) printf("Matrix multiplication on %d blocks x %d threads\n", blockCount, BLOCK_SIZE);
	mulKernel <<< blockCount, BLOCK_SIZE >>>(a.cols, b.cols, b.cols * a.rows, a.mat, b.mat, mat);
	cudaDeviceSynchronize();
	return Matrix(b.cols, a.rows, mat);
}

__global__ void addKernel(const int n, double* A, double* B, double* C)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		C[j] = A[j] + B[j];
	}
}

void Matrix::refAdd(Matrix& result, const Matrix& a, const Matrix& b)
{
	int blockCount = (a.cols * a.rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
	addKernel <<< blockCount, BLOCK_SIZE >>>(a.cols * a.rows, a.mat, b.mat, result.mat);
	cudaDeviceSynchronize();
}

Matrix operator+(const Matrix& a, const Matrix& b)
{
	if (a.cols != b.cols || a.rows != b.rows) throw "dimensions must equal for addition";
	double* mat;
	cudaMallocManaged(&mat, a.cols * a.rows * sizeof(double));
	int blockCount = (a.cols * a.rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
	if (PRINT_LOG) printf("Matrix addition on %d blocks x %d threads\n", blockCount, BLOCK_SIZE);
	addKernel <<< blockCount, BLOCK_SIZE >>>(a.cols * a.rows, a.mat, b.mat, mat);
	cudaDeviceSynchronize();
	return Matrix(a.cols, a.rows, mat);
}

__global__ void subKernel(const int n, double* A, double* B, double* C)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		C[j] = A[j] - B[j];
	}
}

void Matrix::refSub(Matrix& result, const Matrix& a, const Matrix& b)
{
	int blockCount = (a.cols * a.rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
	subKernel <<< blockCount, BLOCK_SIZE >> >(a.cols * a.rows, a.mat, b.mat, result.mat);
	cudaDeviceSynchronize();
}

Matrix operator-(const Matrix& a, const Matrix& b)
{
	if (a.cols != b.cols || a.rows != b.rows) throw "dimensions must equal for addition";
	double* mat;
	cudaMallocManaged(&mat, a.cols * a.rows * sizeof(double));
	int blockCount = (a.cols * a.rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
	if (PRINT_LOG) printf("Matrix addition on %d blocks x %d threads\n", blockCount, BLOCK_SIZE);
	subKernel <<< blockCount, BLOCK_SIZE >>>(a.cols * a.rows, a.mat, b.mat, mat);
	cudaDeviceSynchronize();
	return Matrix(a.cols, a.rows, mat);
}

__global__ void additiveInverseKernel(const int n, double* A, double* B)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int j = index; j < n; j += stride)
	{
		A[j] = -B[j];
	}
}

Matrix operator-(const Matrix& a)
{
	double* mat;
	cudaMallocManaged(&mat, a.cols * a.rows * sizeof(double));
	int blockCount = (a.cols * a.rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
	additiveInverseKernel <<<blockCount, BLOCK_SIZE >>>(a.cols * a.rows, mat, a.mat);
	cudaDeviceSynchronize();
	return Matrix(a.cols, a.rows, mat);
}
