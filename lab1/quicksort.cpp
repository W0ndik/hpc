#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// ������� ������� ����������
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

// ������� ��� ������� ���� ��������������� ������ �������
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

    const int N = 1000000; // ������ �������
    const int RANGE = 1000000; // �������� �����
    std::vector<int> global_array;
    std::vector<int> local_array;

    double start_time, end_time;

    if (rank == 0) {
        // ��������� ������� �� ������� ��������
        global_array.resize(N);
        //std::srand(static_cast<unsigned>(std::time(0)));
        std::srand(0);
        for (int i = 0; i < N; ++i) {
            global_array[i] = std::rand() % RANGE; // ����� �� 0 �� RANGE
        }
        std::cout << "Processor " << processor_name << ", rank " << rank
                  << ": Generated an array of size " << N << ".\n";
    }

    // ������������� ������� ����� ����������
    int local_size = N / size;
    local_array.resize(local_size);

    start_time = MPI_Wtime();

    MPI_Scatter(global_array.data(), local_size, MPI_INT,
                local_array.data(), local_size, MPI_INT,
                0, MPI_COMM_WORLD);

    std::cout << "Processor " << processor_name << ", rank " << rank
              << " out of " << size << " processors: Received array for sorting.\n";

    // ���������� ���������� �������
    quicksort(local_array, 0, local_size - 1);

    // ������������� ������� ��������
    int step = 1;
    while (step < size) {
        if (rank % (2 * step) == 0) {
            if (rank + step < size) {
                // ��������� ���������������� ������� �� ������
                int recv_size;
                MPI_Recv(&recv_size, 1, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<int> neighbor_array(recv_size);
                MPI_Recv(neighbor_array.data(), recv_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNOR>
                // ������� ��������
                std::vector<int> merged_array;
                merge_arrays(local_array, neighbor_array, merged_array);
                local_array.swap(merged_array);
            }
        } else if (rank % step == 0) {
            int dest = rank - step;

            // �������� ������ ������� ������
            int send_size = local_array.size();
            MPI_Send(&send_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(local_array.data(), send_size, MPI_INT, dest, 0, MPI_COMM_WORLD);
            break; // ����� �������� ������ �������� ���������
        }
        step *= 2;
    }

    // ����������� �� ��������� �������
    if (rank == 0) {
        std::cout << "Processor " << processor_name << ", rank " << rank
                  << ": Finished merging sorted arrays.\n";
    }

    // ������������� ��������� ����� ������� �������
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // ����� ������� ����������
    if (rank == 0) {
        std::cout << "Execution time: " << end_time - start_time << " seconds.\n";
    }

    MPI_Finalize();
    return 0;
}