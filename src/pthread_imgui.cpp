#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <pthread_barrier.h>
#include <pthread.h>
#include <vector>

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

static float gravity = 100;
static float space = 800;
static float radius = 5;
static int bodies = 200;
static float elapse = 0.02;
static float max_mass = 50;
static int thread_num = 10;
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

void schedule(int thread_num)
{
    std::vector<pthread_t> threads(thread_num);
    pthread_barrier_init(&barrier, NULL, thread_num);
    pool.clear_acceleration();
    pool.clear_delta_vector();
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
                    ImGui::DragInt("ThreadNumber", &thread_num, 1, 1, 2000, "%d");
                    ImGui::ColorEdit4("Color", &color.x);
                    if (current_space != space || current_bodies != bodies || current_max_mass != max_mass)
                    {
                        space = current_space;
                        bodies = current_bodies;
                        max_mass = current_max_mass;
                        pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};
                    }
                    {
                        const ImVec2 p = ImGui::GetCursorScreenPos();
                        schedule(thread_num);
                        for (size_t i = 0; i < pool.size(); ++i)
                        {
                            auto body = pool.get_body(i);
                            auto x = p.x + static_cast<float>(body.get_x());
                            auto y = p.y + static_cast<float>(body.get_y());
                            draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
                        }
                    }
                    ImGui::End();
                });
}
