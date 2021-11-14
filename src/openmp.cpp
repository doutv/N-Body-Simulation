#include <nbody/body.hpp>
#include <vector>
#include <omp.h>
#include <iostream>

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

static float gravity = 100;
static float space = 800;
static float radius = 5;
static int bodies = 200;
static float elapse = 0.1;
static float max_mass = 50;
BodyPool pool(static_cast<size_t>(bodies), space, max_mass);

void schedule()
{
    pool.clear_acceleration();
    pool.init_delta_vector();
#pragma omp parallel for shared(pool)
    for (size_t i = 0; i < pool.size(); ++i)
    {
        for (size_t j = 0; j < pool.size(); ++j)
        {
            if (i == j)
                continue;
            pool.shared_memory_check_and_update(pool.get_body(i), pool.get_body(j), radius, gravity);
        }
    }
#pragma omp parallel for shared(pool)
    for (size_t i = 0; i < pool.size(); ++i)
    {
        pool.get_body(i).update_by_delta_vector();
        pool.get_body(i).update_for_tick(elapse, space, radius);
    }
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: openmp <size> <rounds> <thread_num>" << std::endl;
        return 0;
    }
    size_t rounds, thread_num;
    bodies = atoi(argv[1]);
    rounds = atoi(argv[2]);
    thread_num = atoi(argv[3]);
    omp_set_num_threads(thread_num);
    BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
    using namespace std::chrono;
    auto begin = high_resolution_clock::now();
    for (size_t i = 0; i < rounds; i++)
    {
        schedule();
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - begin).count() / rounds;
    std::cout << "thread number: " << thread_num << std::endl;
    std::cout << "problem size: " << pool.size() << std::endl;
    std::cout << "duration(ns/round): " << duration << std::endl;
    std::cout << "rounds: " << rounds << std::endl;
    return 0;
}
