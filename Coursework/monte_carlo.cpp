#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <algorithm>
#include <cstdint> // Для типов uint64_t и uint32_t

// Реализация генератора Соболя
class SobolGenerator {
private:
    int dimension;
    uint64_t count;
    std::vector<uint32_t> direction_vectors;

    void init_direction_vectors() {
        direction_vectors.resize(32);
        for (int i = 0; i < 32; ++i) {
            direction_vectors[i] = 1u << (31 - i);
        }
    }

public:
    SobolGenerator(int dim) : dimension(dim), count(0) {
        if (dimension != 1) {
            throw std::runtime_error("Текущая реализация поддерживает только 1 измерение.");
        }
        init_direction_vectors();
    }

    double next() {
        uint32_t x = 0;
        uint64_t c = count++;
        for (int i = 0; i < 32; ++i) {
            if ((c >> i) & 1) {
                x ^= direction_vectors[i];
            }
        }
        return static_cast<double>(x) / static_cast<double>(UINT32_MAX);
    }
};

// Приближенная обратная функция ошибки через std::erf
double approx_erfinv(double x) {
    double a = 0.147;
    double ln1mx2 = std::log(1 - x * x);
    double part1 = 2 / (M_PI * a) + ln1mx2 / 2.0;

    return std::copysign(
        std::sqrt(std::sqrt(part1 * part1 - ln1mx2 / a) - part1), x);
}

// Преобразование равномерного распределения в нормальное
double uniform_to_normal(double u) {
    double x = 2.0 * u - 1.0;
    return std::sqrt(2.0) * approx_erfinv(x);
}

// Метод Монте-Карло для оценки стоимости опциона
double monte_carlo_option_price(double S0, double K, double T, double r, double sigma, int n_simulations, unsigned seed, int rank, int size) {
    SobolGenerator sobol(1);
    for (int i = 0; i < seed + rank; ++i)
        sobol.next();

    double payoff_sum = 0.0;
    for (int i = rank; i < n_simulations; i += size) {
        double u = sobol.next();
        double z = uniform_to_normal(u);
        double ST = S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * z * std::sqrt(T));
        payoff_sum += std::max(0.0, ST - K);
    }

    double payoff_avg = payoff_sum / n_simulations;
    return payoff_avg * std::exp((-r) * T);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    std::cout << "Hello world from processor " << processor_name << ", rank " << rank << " out of " << size << " processors\n";

    double S0 = 100.0;     // Начальная цена актива
    double K = 110.0;      // Цена исполнения
    double T = 10.0;        // Время до исполнения
    double r = 0.2;       // Безрисковая ставка
    double sigma = 0.2;    // Волатильность
    int n_simulations = 100000; // Количество симуляций
    unsigned seed = 67890; // Начальный seed

    // Измерение времени выполнения
    double start_time = MPI_Wtime();

    double local_price = monte_carlo_option_price(S0, K, T, r, sigma, n_simulations, seed, rank, size);

    std::cout << local_price << " is a local price in processor " << "\n";


    double global_price = 0.0;
    MPI_Reduce(&local_price, &global_price, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Option price (Monte Carlo with Sobol): " << global_price << "\n";
        std::cout << "Execution time: " << end_time - start_time << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
 
