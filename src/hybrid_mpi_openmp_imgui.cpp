#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <mpi.h>

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
BodyPool pool(static_cast<size_t>(bodies), space, max_mass);

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
    int elements_per_process = pool.size() / world_size;
    size_t st_idx = elements_per_process * rank;
    size_t end_idx = st_idx + elements_per_process;
#ifdef DEBUG
    printf("rank %d: st_idx %zu: end_idx %zu\n", rank, st_idx, end_idx);
#endif
    if (rank == world_size - 1)
        end_idx = pool.size();

    pool.clear_acceleration();
    pool.clear_delta_vector();
#pragma omp parallel for shared(pool)
    for (size_t i = st_idx; i < end_idx; i++)
    {
        for (size_t j = 0; j < pool.size(); j++)
        {
            if (i == j)
                continue;
            pool.shared_memory_check_and_update(pool.get_body(i), pool.get_body(j), local_radius, local_gravity);
        }
    }
    // Step 2
#pragma omp barrier
#pragma omp parallel for shared(pool)
    for (size_t i = st_idx; i < end_idx; i++)
    {
        pool.get_body(i).update_by_delta_vector();
        pool.get_body(i).update_for_tick(local_elapse, local_space, local_radius);
    }
    // Gather bodies data
    std::vector<int> recvcounts;
    std::vector<int> displs;
    int sendcount = end_idx - st_idx;
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
    printf("rank: %d\n", rank);
    printVector(pool.vx);
    printVector(pool.vy);
    printVector(pool.ax);
    printVector(pool.ay);
    // printVector(recvcounts);
    // printVector(displs);
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
}

int main(int argc, char **argv)
{
    int rank, world_size;
    const int rounds = 1000;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    UNUSED(argc, argv);
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    if (rank == 0)
    {
        int cur_rounds = 0;
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
                        ImGui::DragInt("Bodies", &current_bodies, 1, 2, 1000, "%d");
                        ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
                        ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
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
                            worker(rank, world_size);
                            for (size_t i = 0; i < pool.size(); ++i)
                            {
                                auto body = pool.get_body(i);
                                auto x = p.x + static_cast<float>(body.get_x());
                                auto y = p.y + static_cast<float>(body.get_y());
                                draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
                            }
                            ++cur_rounds;
                            if (cur_rounds == rounds)
                            {
                                context->quit();
                            }
                        }
                        ImGui::End();
                    });
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
