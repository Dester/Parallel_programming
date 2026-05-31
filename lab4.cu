#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <cuda_runtime.h>

bool readMatrix(const std::string& filename, std::vector<std::vector<double>>& matrix, int N) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
        return false;
    }

    matrix.resize(N, std::vector<double>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (!(infile >> matrix[i][j])) {
                std::cerr << "Ошибка: недостаточно данных в файле " << filename << std::endl;
                return false;
            }
        }
    }
    infile.close();
    return true;
}

bool writeMatrix(const std::string& filename, const std::vector<std::vector<double>>& matrix, int N) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Ошибка: не удалось создать файл " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outfile << matrix[i][j] << (j == N - 1 ? "" : " ");
        }
        outfile << "\n";
    }
    outfile.close();
    return true;
}

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void multiplyMatrix(const std::string& A_, const std::string& B_, const std::string& C_, int N, int blockX = 16, int blockY = 16) {
    auto start_total = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> A2D, B2D;
    std::vector<std::vector<double>> C2D(N, std::vector<double>(N, 0.0));

    if (!readMatrix(A_, A2D, N) || !readMatrix(B_, B2D, N)) {
        return;
    }

    std::vector<double> A_flat(N * N);
    std::vector<double> B_flat(N * N);
    std::vector<double> C_flat(N * N, 0.0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A_flat[i * N + j] = A2D[i][j];
            B_flat[i * N + j] = B2D[i][j];
        }
    }

    double* d_A, * d_B, * d_C;
    size_t bytes = N * N * sizeof(double);

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, A_flat.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(blockX, blockY);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    auto start_gpu = std::chrono::high_resolution_clock::now();

    matrixMulKernel << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(C_flat.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C2D[i][j] = C_flat[i * N + j];
        }
    }

    writeMatrix(C_, C2D, N);

    auto end_total = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration_total = end_total - start_total;
    std::chrono::duration<double, std::milli> duration_gpu = end_gpu - start_gpu;

    std::cout << "Размер: " << N << "x" << N << " | Конфигурация блока: " << blockX << "x" << blockY << "\n"
        << "  -> Время вычислений (GPU): " << duration_gpu.count() << " мс\n"
        << "  -> Общее время (с I/O): " << duration_total.count() << " мс\n" << std::endl;
}

int main() {
    std::setlocale(LC_ALL, "Russian");

    multiplyMatrix("input/200A.txt", "input/200B.txt", "output/200C.txt", 200, 8, 8);
    multiplyMatrix("input/400A.txt", "input/400B.txt", "output/400C.txt", 400, 8, 8);
    multiplyMatrix("input/800A.txt", "input/800B.txt", "output/800C.txt", 800, 8, 8);
    multiplyMatrix("input/1200A.txt", "input/1200B.txt", "output/1200C.txt", 1200, 8, 8);
    multiplyMatrix("input/1600A.txt", "input/1600B.txt", "output/1600C.txt", 1600, 8, 8);
    multiplyMatrix("input/2000A.txt", "input/2000B.txt", "output/2000C.txt", 2000, 8, 8);

    return 0;
}