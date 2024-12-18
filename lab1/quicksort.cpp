#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// Функция быстрой сортировки
void quicksort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; ++j) {
            if (arr[j] < pivot) {
                ++i;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[high]);
        int p = i + 1;

        quicksort(arr, low, p - 1);
        quicksort(arr, p + 1, high);
    }
}

// Функция для слияния двух отсортированных частей массива
void merge_arrays(const std::vector<int>& left, const std::vector<int>& right, std::vector<int>& result) {
    size_t i = 0, j = 0, k = 0;
    result.resize(left.size() + right.size());

    while (i < left.size() && j < right.size()) {
        if (left[i] < right[j]) {
            result[k++] = left[i++];
        } else {
            result[k++] = right[j++];
        }
    }

    while (i < left.size()) {
        result[k++] = left[i++];
    }

    while (j < right.size()) {
        result[k++] = right[j++];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(processor_name, &name_len);

    const int N = 1000000; // Размер массива
    const int RANGE = 1000000; // Диапазон чисел
    std::vector<int> global_array;
    std::vector<int> local_array;

    double start_time, end_time;

    if (rank == 0) {
        // Генерация массива на главном процессе
        global_array.resize(N);
        //std::srand(static_cast<unsigned>(std::time(0)));
        std::srand(0);
        for (int i = 0; i < N; ++i) {
            global_array[i] = std::rand() % RANGE; // Числа от 0 до RANGE
        }
        std::cout << "Processor " << processor_name << ", rank " << rank
                  << ": Generated an array of size " << N << ".\n";
    }

    // Распределение массива между процессами
    int local_size = N / size;
    local_array.resize(local_size);

    start_time = MPI_Wtime();

    MPI_Scatter(global_array.data(), local_size, MPI_INT,
                local_array.data(), local_size, MPI_INT,
                0, MPI_COMM_WORLD);

    std::cout << "Processor " << processor_name << ", rank " << rank
              << " out of " << size << " processors: Received array for sorting.\n";

    // Сортировка локального массива
    quicksort(local_array, 0, local_size - 1);

    // Иерархическое слияние массивов
    int step = 1;
    while (step < size) {
        if (rank % (2 * step) == 0) {
            if (rank + step < size) {
                // Получение отсортированного массива от соседа
                int recv_size;
                MPI_Recv(&recv_size, 1, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<int> neighbor_array(recv_size);
                MPI_Recv(neighbor_array.data(), recv_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNOR>
                // Слияние массивов
                std::vector<int> merged_array;
                merge_arrays(local_array, neighbor_array, merged_array);
                local_array.swap(merged_array);
            }
        } else if (rank % step == 0) {
            int dest = rank - step;

            // Отправка своего массива соседу
            int send_size = local_array.size();
            MPI_Send(&send_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(local_array.data(), send_size, MPI_INT, dest, 0, MPI_COMM_WORLD);
            break; // После отправки работа процесса завершена
        }
        step *= 2;
    }

    // Уведомление об окончании слияния
    if (rank == 0) {
        std::cout << "Processor " << processor_name << ", rank " << rank
                  << ": Finished merging sorted arrays.\n";
    }

    // Синхронизация процессов перед замером времени
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Вывод времени выполнения
    if (rank == 0) {
        std::cout << "Execution time: " << end_time - start_time << " seconds.\n";
    }

    MPI_Finalize();
    return 0;
}