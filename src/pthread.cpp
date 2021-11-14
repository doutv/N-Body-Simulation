#include <nbody/body.hpp>
#include <pthread.h>
#include <vector>
#include <iostream>
#include <chrono>

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

static float gravity = 100;
static float space = 800;
static float radius = 5;
static int bodies = 200;
static float elapse = 0.02;
static float max_mass = 50;
BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
pthread_barrier_t barrier;

struct idx_struct
{
    size_t st_idx, end_idx;
};

void *worker(void *data)
{
    struct idx_struct *p = reinterpret_cast<idx_struct *>(data);
    for (size_t i = p->st_idx; i < p->end_idx; i++)
    {
        for (size_t j = 0; j < pool.size(); ++j)
        {
            if (i == j)
                continue;
            pool.shared_memory_check_and_update(pool.get_body(i), pool.get_body(j), radius, gravity);
        }
    }
    pthread_barrier_wait(&barrier);
    for (size_t i = p->st_idx; i < p->end_idx; i++)
    {
        pool.get_body(i).update_by_delta_vector();
        pool.get_body(i).update_for_tick(elapse, space, radius);
    }
    return nullptr;
}
void schedule(size_t thread_num)
{
    std::vector<pthread_t> threads(thread_num);
    pthread_barrier_init(&barrier, NULL, thread_num);
    pool.clear_acceleration();
    pool.init_delta_vector();
    size_t idx_per_thread = pool.size() / thread_num;
    size_t remainder = pool.size() % thread_num;
    size_t st_idx = 0;
    std::vector<idx_struct> idx_struct_arr;
    for (size_t i = 0; i < threads.size(); i++)
    {
        size_t end_idx = i < remainder ? st_idx + idx_per_thread + 1 : st_idx + idx_per_thread;
        idx_struct_arr.push_back({st_idx, end_idx});
        st_idx = end_idx;
    }
    for (size_t i = 0; i < threads.size(); i++)
    {
        pthread_create(&threads[i], nullptr, worker, reinterpret_cast<void *>(&idx_struct_arr[i]));
    }
    for (auto &thread : threads)
    {
        pthread_join(thread, nullptr);
    }
}

int main(int argc, char **argv)
{
    UNUSED(argc, argv);
    if (argc < 3)
    {
        std::cout << "Usage: pthread <size> <rounds> <thread_num>" << std::endl;
        return 0;
    }
    size_t rounds, thread_num;
    bodies = atoi(argv[1]);
    rounds = atoi(argv[2]);
    thread_num = atoi(argv[3]);
    BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
    using namespace std::chrono;
    auto begin = high_resolution_clock::now();
    for (size_t i = 0; i < rounds; i++)
    {
        schedule(thread_num);
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - begin).count() / rounds;
    std::cout << "thread number: " << thread_num << std::endl;
    std::cout << "problem size: " << pool.size() << std::endl;
    std::cout << "duration(ns/round): " << duration << std::endl;
    std::cout << "rounds: " << rounds << std::endl;
    return 0;
}
