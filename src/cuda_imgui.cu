#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/cuda_body.hpp>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

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
    printf("threadIdx: %zd\n", i);
#endif
    for (size_t j = 0; j < pool->size; ++j)
    {
        if (i == j)
            continue;
        pool->shared_memory_check_and_update(pool->get_body(i), pool->get_body(j), radius, gravity);
    }
    // barrier
    pool->get_body(i).update_by_delta_var();
    pool->get_body(i).update_for_tick(elapse, space, radius);
}

__host__ void schedule()
{
    pool->clear_acceleration();
    dim3 grid(1);
    dim3 block(pool->size);
    worker<<<grid, block>>>();
    cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
    UNUSED(argc, argv);
    pool = new BodyPool(static_cast<size_t>(bodies), space, max_mass);
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;
    graphic::GraphicContext context{"Assignment 3"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *)
                {
                    auto io = ImGui::GetIO();
                    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
                    ImGui::SetNextWindowSize(io.DisplaySize);
                    ImGui::Begin("Assignment 3", nullptr,
                                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
                    ImDrawList *draw_list = ImGui::GetWindowDrawList();
                    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                                ImGui::GetIO().Framerate);
                    ImGui::DragFloat("Space", &current_space, 10, 200, 1600, "%f");
                    ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
                    ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
                    ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
                    ImGui::DragInt("Bodies", &current_bodies, 1, 2, 2000, "%d");
                    ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
                    ImGui::ColorEdit4("Color", &color.x);
                    if (current_space != space || current_bodies != bodies || current_max_mass != max_mass)
                    {
                        space = current_space;
                        bodies = current_bodies;
                        max_mass = current_max_mass;
                        delete (pool);
                        pool = new BodyPool{static_cast<size_t>(bodies), space, max_mass};
                    }
                    {
                        const ImVec2 p = ImGui::GetCursorScreenPos();
                        schedule();
                        for (size_t i = 0; i < pool->size; ++i)
                        {
                            auto body = pool->get_body(i);
                            auto x = p.x + static_cast<float>(body.get_x());
                            auto y = p.y + static_cast<float>(body.get_y());
                            draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
                        }
                    }
                    ImGui::End();
                });
    delete pool;
    cudaDeviceReset();
    return 0;
}
