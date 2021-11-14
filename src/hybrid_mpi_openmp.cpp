#include <nbody/body.hpp>
#include <mpi.h>
#include <iostream>
#include <omp.h>
#include <chrono>

// #define DEBUG

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

template <typename Container>
void printVector(const Container &cont)
{
    for (auto const &x : cont)
    {
        std::cout << x << " ";
    }
    std::cout << '\n';
}

static float gravity = 100;
static float space = 800;
static float radius = 5;
static int bodies = 200;
static float elapse = 0.02;
static float max_mass = 50;
BodyPool pool(static_cast<size_t>(bodies));

void worker(int rank, int world_size)
{
    double local_elapse;
    double local_gravity;
    double local_space;
    double local_radius;
    double local_max_mass;
    int local_bodies;
    if (rank == 0)
    {
        local_elapse = elapse;
        local_gravity = gravity;
        local_space = space;
        local_radius = radius;
        local_max_mass = max_mass;
        local_bodies = bodies;
    }
    MPI_Bcast(&local_elapse, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_gravity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_space, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_radius, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_max_mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_bodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0)
    {
        pool = BodyPool{static_cast<size_t>(local_bodies), local_space, local_max_mass};
    }

    MPI_Bcast(pool.x.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.y.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.vx.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.vy.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.ax.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.ay.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.m.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 1
    pool.clear_acceleration();
    int elements_per_process = pool.size() / world_size;
    size_t st_idx = elements_per_process * rank;
    size_t end_idx = st_idx + elements_per_process;
    if (rank == world_size - 1)
        end_idx = pool.size();

#pragma omp parallel for shared(pool)
    for (size_t i = st_idx; i < end_idx; i++)
    {
        for (size_t j = 0; j < pool.size(); j++)
        {
            if (i == j)
                continue;
            pool.mpi_check_and_update(pool.get_body(i), pool.get_body(j), local_radius, local_gravity);
        }
    }
#pragma omp parallel for shared(pool)
    // Step 2
    for (size_t i = st_idx; i < end_idx; i++)
    {
        pool.get_body(i).update_for_tick(local_elapse, local_space, local_radius);
    }
    // Gather bodies data
    std::vector<int> recvcounts;
    std::vector<int> displs;
    int sendcount = end_idx - st_idx;
#ifdef DEBUG
    printf("rank %d: st_idx: %zu ; end_idx: %zu ; sendcount: %d\n", rank, st_idx, end_idx, sendcount);
#endif
    if (rank == 0)
    {
        recvcounts.resize(world_size, elements_per_process);
        recvcounts.back() += pool.size() % world_size;
        displs.resize(world_size, 0);
        for (size_t i = 1; i < displs.size(); i++)
        {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }
#ifdef DEBUG
    printf("rank %d pool:\n", rank);
    printVector(pool.x);
    printVector(pool.y);
    printVector(pool.vx);
    printVector(pool.vy);
    printVector(pool.ax);
    printVector(pool.ay);
    printf("recvcounts:\n");
    printVector(recvcounts);
    printf("displs:\n");
    printVector(displs);
#endif
    // MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //             void *recvbuf, const int *recvcounts, const int *displs,
    //             MPI_Datatype recvtype, int root, MPI_Comm comm)
    MPI_Gatherv(pool.x.data() + st_idx, sendcount, MPI_DOUBLE, pool.x.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(pool.y.data() + st_idx, sendcount, MPI_DOUBLE, pool.y.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(pool.vx.data() + st_idx, sendcount, MPI_DOUBLE, pool.vx.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(pool.vy.data() + st_idx, sendcount, MPI_DOUBLE, pool.vy.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(pool.ax.data() + st_idx, sendcount, MPI_DOUBLE, pool.ax.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(pool.ay.data() + st_idx, sendcount, MPI_DOUBLE, pool.ay.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#ifdef DEBUG
    printf("rank %d exit worker()\n", rank);
#endif
}

int main(int argc, char **argv)
{
    int rank, world_size;
    int rounds;
    int thread_num;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (argc < 4)
    {
        std::cout << "Usage: hybrid <size> <rounds> <thread_num>" << std::endl;
        return 0;
    }
    bodies = atoi(argv[1]);
    rounds = atoi(argv[2]);
    thread_num = atoi(argv[3]);
    omp_set_num_threads(thread_num);
    if (rank == 0)
    {
        using namespace std::chrono;
        pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
        auto begin = high_resolution_clock::now();
        for (int i = 0; i < rounds; i++)
        {
            worker(rank, world_size);
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end - begin).count() / rounds;
        std::cout << "world size: " << world_size << std::endl;
        std::cout << "thread number: " << thread_num << std::endl;
        std::cout << "problem size: " << pool.size() << std::endl;
        std::cout << "duration(ns/round): " << duration << std::endl;
        std::cout << "rounds: " << rounds << std::endl;
    }
    else
    {
        for (int i = 0; i < rounds; i++)
        {
            worker(rank, world_size);
        }
    }
    MPI_Finalize();
    return 0;
}