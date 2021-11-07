#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <mpi.h>

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

void root_schedule(double elapse,
                   double gravity,
                   double space,
                   double radius,
                   double max_mass,
                   int bodies,
                   BodyPool &pool)
{
    MPI_Bcast(&elapse, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gravity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&space, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&radius, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bodies, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(pool.x.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.y.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.vx.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.vy.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.ax.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.ay.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.m.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 1: merge data
    // MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
    double dx, dy, dvx, dvy, dax, day;
    MPI_Reduce(pool.dx.data(), &dx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(pool.dy.data(), &dy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(pool.dvx.data(), &dvx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(pool.dvy.data(), &dvy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(pool.dax.data(), &dax, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(pool.day.data(), &day, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

void worker(int rank, int world_size, BodyPool &pool)
{
    double elapse;
    double gravity;
    double space;
    double radius;
    double max_mass;
    int bodies;
    MPI_Bcast(&elapse, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gravity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&space, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&radius, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    pool = BodyPool{static_cast<size_t>(bodies), space, max_mass};

    MPI_Bcast(pool.x.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.y.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.vx.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.vy.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.ax.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.ay.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pool.m.data(), pool.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 1
    pool.mpi_init_delta_vector();
    int elements_per_process = pool.size() / world_size;
    int st_idx = elements_per_process * rank;
    int end_idx = st_idx + elements_per_process;
    if (rank == world_size - 1)
        end_idx = pool.size();
    for (int i = st_idx; i < end_idx; i++)
    {
        for (int j = i + 1; j < pool.size(); j++)
        {
            pool.check_and_update(pool.get_body(i), pool.get_body(j), radius, gravity);
        }
    }
    // reduce delta vector to the root process
    // MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
    double dx, dy, dvx, dvy, dax, day;
    MPI_Reduce(pool.dx.data(), &dx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(pool.dy.data(), &dy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(pool.dvx.data(), &dvx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(pool.dvy.data(), &dvy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(pool.dax.data(), &dax, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(pool.day.data(), &day, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // Step 2
    for (int i = st_idx; i < end_idx; i++)
    {
        pool.get_body(i).update_for_tick(elapse, space, radius);
    }
}

int main(int argc, char **argv)
{
    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    UNUSED(argc, argv);
    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 200;
    static float elapse = 0.001;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;
    BodyPool pool(static_cast<size_t>(bodies), space, max_mass);
    if (rank == 0)
    {
        static float current_space = space;
        static float current_max_mass = max_mass;
        static int current_bodies = bodies;
        graphic::GraphicContext context{"Assignment 2"};
        context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *)
                    {
                        auto io = ImGui::GetIO();
                        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
                        ImGui::SetNextWindowSize(io.DisplaySize);
                        ImGui::Begin("Assignment 2", nullptr,
                                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
                        ImDrawList *draw_list = ImGui::GetWindowDrawList();
                        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                                    ImGui::GetIO().Framerate);
                        ImGui::DragFloat("Space", &current_space, 10, 200, 1600, "%f");
                        ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
                        ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
                        ImGui::DragInt("Bodies", &current_bodies, 1, 2, 100, "%d");
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
                            root_schedule(elapse, gravity, space, radius, max_mass, bodies, pool);
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
    else
    {
        while (1)
        {
            worker(rank, world_size, pool);
        }
    }
    MPI_Finalize();
    return 0;
}
