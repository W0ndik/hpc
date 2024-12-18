#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>

using SystemFunction = std::function<double(double, double)>;

void rungeKutta4(
    const std::vector<SystemFunction>& systems,
    const std::vector<double>& y0s,
    double t_start,
    double dt,
    int steps,
    std::vector<std::vector<double>>& results
) {
    size_t num_systems = systems.size();
    results.resize(num_systems, std::vector<double>(steps + 1));

    for (size_t i = 0; i < num_systems; ++i) {
        results[i][0] = y0s[i];
    }

    for (int i = 0; i < steps; ++i) {
        double t = t_start + i * dt;

        for (size_t j = 0; j < num_systems; ++j) {
            double y = results[j][i];
            double k1 = dt * systems[j](t, y);
            double k2 = dt * systems[j](t + 0.5 * dt, y + 0.5 * k1);
            double k3 = dt * systems[j](t + 0.5 * dt, y + 0.5 * k2);
            double k4 = dt * systems[j](t + dt, y + k3);
            results[j][i + 1] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
    	    std::cout <<"Следующее значение y+1 " << j << "-ого д.у. " << results[j][i+1] << "\n";
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    int N = 100;
    // if (rank == 0) {
    //     std::cout << "Введите количество шагов (N): ";
    //     std::cin >> N;
    // }

    // Распространение значения N среди всех процессов
    // MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const double t0 = 0.0;
    const double t_end = 10.0;
    const double dt = (t_end - t0) / N;

    const std::vector<SystemFunction> systems = {
        [](double t, double y) { return 2 * std::sqrt(y) + 2 * y; },
        [](double t, double y) { return 3 * std::sqrt(y) + 3 * y; },
        [](double t, double y) { return 4 * std::sqrt(y) + 4 * y; }
    };

    const std::vector<double> y0s = { 1.0, 1.0, 1.0 };

    int local_steps = N / size;
    double local_t_start = t0 + rank * local_steps * dt;

    // Последний процесс обрабатывает оставшиеся шаги
    if (rank == size - 1) {
        local_steps += N % size;
    }

    std::vector<std::vector<double>> local_results;

    // Основной расчет в каждом процессе
    rungeKutta4(systems, y0s, local_t_start, dt, local_steps, local_results);

    std::vector<std::vector<double>> global_results;
    if (rank == 0) {
        global_results.resize(systems.size(), std::vector<double>(N + 1));
    }

    // Сбор данных всех процессов
    for (size_t i = 0; i < systems.size(); ++i) {

        MPI_Gather(
            local_results[i].data(), local_steps + 1, MPI_DOUBLE,
            rank == 0 ? global_results[i].data() : nullptr,
            local_steps + 1, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );
    }
	
    
    std::cout << "Hello from " << rank <<  "\n";
    //if (rank == 0) {
        //for (size_t step = 0; step <= N; ++step) {
            //double t = t0 + step * dt;
	    
            //std::cout << "t = " << t;
            // for (size_t j = 0; j < systems.size(); ++j) {
            //     std::cout << ", y" << j + 1 << " = " << global_results[j][step];
            // }
            //std::cout << std::endl;
        //}
    //}

    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Время выполнения программы: " << end_time - start_time << " секунд" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
