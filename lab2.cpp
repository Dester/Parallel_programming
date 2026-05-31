#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <clocale>
#include <omp.h>

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

void multiplyMatrix(const std::string& A_, const std::string& B_, const std::string& C_, int N, int num_threads) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> A, B;
    std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));

    if (!readMatrix(A_, A, N) || !readMatrix(B_, B, N)) {
        return;
    }

#pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Время перемножения матриц размера " << N << "x" << N << " для " << num_threads << " потока(ов): " << duration.count() << " мс" << std::endl;
    writeMatrix(C_, C, N);
}

int main() {
    std::setlocale(LC_ALL, "Russian");

    for (int i = 1; i < 10; i *= 2) {
        multiplyMatrix("input/200A.txt", "input/200B.txt", "output/200C.txt", 200, i);
        multiplyMatrix("input/400A.txt", "input/400B.txt", "output/400C.txt", 400, i);
        multiplyMatrix("input/800A.txt", "input/800B.txt", "output/800C.txt", 800, i);
        multiplyMatrix("input/1200A.txt", "input/1200B.txt", "output/1200C.txt", 1200, i);
        multiplyMatrix("input/1600A.txt", "input/1600B.txt", "output/1600C.txt", 1600, i);
        multiplyMatrix("input/2000A.txt", "input/2000B.txt", "output/2000C.txt", 2000, i);
    }
    
    return 0;
}