#include <nbody/cuda_body.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define DEBUG

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

__device__ __managed__ static float gravity = 100;
__device__ __managed__ static float space = 800;
__device__ __managed__ static float radius = 5;
__device__ __managed__ static int bodies = 200;
__device__ __managed__ static float elapse = 0.1;
__device__ __managed__ static float max_mass = 50;
__device__ __managed__ BodyPool *pool;

__global__ void worker()
{
    size_t i = threadIdx.x;
#ifdef DEBUG
    // printf("threadIdx: %d \n", i);
#endif
    for (size_t j = 0; j < pool->size; ++j)
    {
        if (i == j)
            continue;
        pool->shared_memory_check_and_update(pool->get_body(i), pool->get_body(j), radius, gravity);
    }
    __syncthreads();
    pool->get_body(i).update_by_delta_var();
    pool->get_body(i).update_for_tick(elapse, space, radius);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: cuda <size> <rounds>" << std::endl;
        return 0;
    }
    size_t rounds;
    bodies = atoi(argv[1]);
    rounds = atoi(argv[2]);
    using namespace std::chrono;
    pool = new BodyPool(static_cast<size_t>(bodies), space, max_mass);
    dim3 grid(1);
    dim3 block(pool->size);
    auto begin = high_resolution_clock::now();
    for (size_t i = 0; i < rounds; i++)
    {
        pool->clear_acceleration();
        worker<<<grid, block>>>();
        cudaDeviceSynchronize();
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - begin).count() / rounds;
    std::cout << "block size: " << grid.x << std::endl;
    std::cout << "threads per block: " << block.x << std::endl;
    std::cout << "problem size: " << pool->size << std::endl;
    std::cout << "duration(ns/round): " << duration << std::endl;
    std::cout << "rounds: " << rounds << std::endl;
#ifdef DEBUG
    printf("pool size: %zd\n", pool->size);
    for (auto &each : pool->x)
        std::cout << each << " ";
    std::cout << std::endl;
    for (auto &each : pool->vx)
        std::cout << each << " ";
    std::cout << std::endl;
    for (auto &each : pool->ax)
        std::cout << each << " ";
    std::cout << std::endl;
#endif
    delete pool;
    cudaDeviceReset();
    return 0;
}
