#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Номер текущего процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Общее количество процессов

    // Вывод номера и общего числа процессов
    std::cout << "Процесс " << rank << " из " << size << " процессов" << std::endl;

    const int M = 20000;  // Число строк матрицы
    const int N = 20000;  // Число столбцов матрицы

    // Матрица и вектор определяем на основном процессе
    std::vector<int> A;
    std::vector<int> x(N, 1);  // Пример вектора x (заполняем единицами)
    std::vector<int> y(M, 0);  // Результат умножения
    
    if (rank == 0) {
        // Генерируем случайную матрицу A на главном процессе
        A.resize(M * N);
        for (int i = 0; i < M * N; ++i) {
            A[i] = std::rand() % 100;  // Случайные числа от 0 до 99
        }
    }

    // Определяем количество строк, которые обрабатывает каждый процесс
    int rows_per_process = M / size;
    std::cout << "Число строк матрицы для каждого процесса " << rows_per_process << std::endl;

    double start_time = MPI_Wtime();

    std::vector<int> local_A(rows_per_process * N);  // Локальная часть матрицы
    std::vector<int> local_y(rows_per_process, 0);   // Локальный результат

    // Разделяем матрицу A по процессам
    MPI_Scatter(A.data(), rows_per_process * N, MPI_INT,
                local_A.data(), rows_per_process * N, MPI_INT, 
                0, MPI_COMM_WORLD);

    // Рассылаем вектор x всем процессам
    MPI_Bcast(x.data(), N, MPI_INT, 0, MPI_COMM_WORLD);

    // Локальное умножение матрицы на вектор
    for (int i = 0; i < rows_per_process; ++i) {
        for (int j = 0; j < N; ++j) {
            local_y[i] += local_A[i * N + j] * x[j];
        }
    }

    // Собираем части результата с каждого процесса в итоговый вектор y
    MPI_Gather(local_y.data(), rows_per_process, MPI_INT,
            y.data(), rows_per_process, MPI_INT, 
            0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    bool output = False;
    if (output == True) {
        // Вывод результата на главном процессе
        if (rank == 0) {
            std::cout << "Результат умножения матрицы на вектор:" << std::endl;
            for (int i = 0; i < M; ++i) {
                std::cout << y[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << "Время выполнения программы: " << end_time - start_time << " секунд" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
