#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <mpi.h>

bool readMatrix(const std::string& filename, std::vector<double>& matrix, int N) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл " << filename << std::endl;
        return false;
    }

    matrix.resize(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (!(infile >> matrix[i * N + j])) {
                std::cerr << "Ошибка: недостаточно данных в файле " << filename << std::endl;
                return false;
            }
        }
    }
    infile.close();
    return true;
}

bool writeMatrix(const std::string& filename, const std::vector<double>& matrix, int N) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Ошибка: не удалось создать файл " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outfile << matrix[i * N + j] << (j == N - 1 ? "" : " ");
        }
        outfile << "\n";
    }
    outfile.close();
    return true;
}

void multiplyMatrix(const std::string& A_, const std::string& B_, const std::string& C_, int N, int rank, int size) {
    std::vector<double> A, B, C;
    B.resize(N * N);

    double start_time;

    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);
        readMatrix(A_, A, N);
        readMatrix(B_, B, N);
        start_time = MPI_Wtime();
    }


    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int offset = 0;

    for (int i = 0; i < size; ++i) {
        int rows = N / size + (i < N % size ? 1 : 0);
        sendcounts[i] = rows * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int local_rows = sendcounts[rank] / N;
    std::vector<double> local_A(sendcounts[rank]);
    std::vector<double> local_C(sendcounts[rank], 0.0);

    MPI_Scatterv(rank == 0 ? A.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_A.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }

    MPI_Gatherv(local_C.data(), sendcounts[rank], MPI_DOUBLE,
        rank == 0 ? C.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        double end_time = MPI_Wtime();
        double duration_ms = (end_time - start_time) * 1000.0;
        std::cout << "Время перемножения (MPI, потоков: " << size << ") матриц " << N << "x" << N << ": " << duration_ms << " мс" << std::endl;
        writeMatrix(C_, C, N);
    }
}

int main(int argc, char** argv) {
    setlocale(LC_ALL, "Russian");

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        multiplyMatrix("input/200A.txt", "input/200B.txt", "output/200C.txt", 200, rank, size);
        multiplyMatrix("input/400A.txt", "input/400B.txt", "output/400C.txt", 400, rank, size);
        multiplyMatrix("input/800A.txt", "input/800B.txt", "output/800C.txt", 800, rank, size);
        multiplyMatrix("input/1200A.txt", "input/1200B.txt", "output/1200C.txt", 1200, rank, size);
        multiplyMatrix("input/1600A.txt", "input/1600B.txt", "output/1600C.txt", 1600, rank, size);
        multiplyMatrix("input/2000A.txt", "input/2000B.txt", "output/2000C.txt", 2000, rank, size);
    }

    MPI_Finalize();

    return 0;
}