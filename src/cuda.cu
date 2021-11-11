#include <cstring>
#include <nbody/body.hpp>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

static float gravity = 100;
static float space = 800;
static float radius = 5;
static int bodies = 200;
static float elapse = 0.1;
static float max_mass = 50;
BodyPool pool(static_cast<size_t>(bodies), space, max_mass);

int main(int argc, char **argv)
{
    UNUSED(argc, argv);
    int dev = 0;
    cudaDeviceProp devProp;
    // cudaError_t cudaError;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM number:" << devProp.multiProcessorCount << std::endl;
    std::cout << "sharedMemPerBlock:" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "maxThreadsPerBlock:" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "maxThreadsPerMultiProcessor:" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "maxThreadsPerMultiProcessor:" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    return 0;
}
