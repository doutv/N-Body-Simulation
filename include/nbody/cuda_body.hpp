//
// Created by schrodinger on 11/2/21.
//
#pragma once

#include <cuda_runtime.h>
#include <random>

class Managed
{
public:
    __host__ void *operator new(size_t len)
    {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    __host__ void operator delete(void *ptr)
    {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

class BodyPool : public Managed
{
public:
    // provides in this way so that
    // it is easier for you to send a the vector with MPI
    static const size_t max_size = 2000;
    double x[max_size];
    double y[max_size];
    double vx[max_size];
    double vy[max_size];
    double ax[max_size];
    double ay[max_size];
    double m[max_size];
    // so the movements of bodies are calculated discretely.
    // if after the collision, we do not separate the bodies a little bit, it may
    // results in strange outcomes like infinite acceleration.
    // hence, we will need to set up a ratio for separation.
    static constexpr double COLLISION_RATIO = 0.01;
    size_t size;

    class Body
    {
        size_t index;
        BodyPool &pool;

        friend class BodyPool;

        __device__ __host__ Body(size_t index, BodyPool &pool) : index(index), pool(pool) { init_delta_var(); }

    public:
        double dx, dy, dvx, dvy;
        __device__ __host__ void init_delta_var()
        {
            dx = dy = dvx = dvy = 0;
        }

        __device__ __host__ double &get_x()
        {
            return pool.x[index];
        }

        __device__ __host__ double &get_y()
        {
            return pool.y[index];
        }

        __device__ __host__ double &get_vx()
        {
            return pool.vx[index];
        }

        __device__ __host__ double &get_vy()
        {
            return pool.vy[index];
        }

        __device__ __host__ double &get_ax()
        {
            return pool.ax[index];
        }

        __device__ __host__ double &get_ay()
        {
            return pool.ay[index];
        }

        __device__ __host__ double &get_m()
        {
            return pool.m[index];
        }

        __device__ __host__ double distance_square(Body &that)
        {
            auto delta_x = get_x() - that.get_x();
            auto delta_y = get_y() - that.get_y();
            return delta_x * delta_x + delta_y * delta_y;
        }

        __device__ __host__ double distance(Body &that)
        {
            return std::sqrt(distance_square(that));
        }

        __device__ __host__ double delta_x(Body &that)
        {
            return get_x() - that.get_x();
        }

        __device__ __host__ double delta_y(Body &that)
        {
            return get_y() - that.get_y();
        }

        __device__ __host__ bool collide(Body &that, double radius)
        {
            return distance_square(that) <= radius * radius;
        }

        // collision with wall
        __device__ __host__ void handle_wall_collision(double position_range, double radius)
        {
            bool flag = false;
            if (get_x() <= radius)
            {
                flag = true;
                get_x() = radius + radius * COLLISION_RATIO;
                get_vx() = -get_vx();
            }
            else if (get_x() >= position_range - radius)
            {
                flag = true;
                get_x() = position_range - radius - radius * COLLISION_RATIO;
                get_vx() = -get_vx();
            }

            if (get_y() <= radius)
            {
                flag = true;
                get_y() = radius + radius * COLLISION_RATIO;
                get_vy() = -get_vy();
            }
            else if (get_y() >= position_range - radius)
            {
                flag = true;
                get_y() = position_range - radius - radius * COLLISION_RATIO;
                get_vy() = -get_vy();
            }
            if (flag)
            {
                get_ax() = 0;
                get_ay() = 0;
            }
        }

        __device__ __host__ void update_for_tick(
            double elapse,
            double position_range,
            double radius)
        {
            get_vx() += get_ax() * elapse;
            get_vy() += get_ay() * elapse;
            handle_wall_collision(position_range, radius);
            get_x() += get_vx() * elapse;
            get_y() += get_vy() * elapse;
            handle_wall_collision(position_range, radius);
        }

        __device__ __host__ void update_by_delta_var()
        {
            get_x() += dx;
            get_y() += dy;
            get_vx() += dvx;
            get_vy() += dvy;
        }
    };

    __host__ BodyPool(size_t size, double position_range, double mass_range) : size(size)
    {
        std::random_device device;
        std::default_random_engine engine{device()};
        std::uniform_real_distribution<double> position_dist{0, position_range};
        std::uniform_real_distribution<double> mass_dist{0, mass_range};
        for (size_t i = 0; i < size; ++i)
        {
            x[i] = position_dist(engine);
        }
        for (size_t i = 0; i < size; ++i)
        {
            y[i] = position_dist(engine);
        }
        for (size_t i = 0; i < size; ++i)
        {
            m[i] = mass_dist(engine);
        }
    }

    __device__ __host__ Body get_body(size_t index)
    {
        return {index, *this};
    }

    __device__ __host__ void clear_acceleration()
    {
        for (size_t i = 0; i < size; i++)
        {
            ax[i] = 0;
            ay[i] = 0;
        }
    }

    // Only update Body i
    __device__ __host__ static void shared_memory_check_and_update(Body i, Body j, double radius, double gravity)
    {
        auto delta_x = i.delta_x(j);
        auto delta_y = i.delta_y(j);
        auto distance_square = i.distance_square(j);
        auto ratio = 1 + COLLISION_RATIO;
        if (distance_square < radius * radius)
        {
            distance_square = radius * radius;
        }
        auto distance = i.distance(j);
        if (distance < radius)
        {
            distance = radius;
        }
        if (i.collide(j, radius))
        {
            auto dot_prod = delta_x * (i.get_vx() - j.get_vx()) + delta_y * (i.get_vy() - j.get_vy());
            auto scalar = 2 / (i.get_m() + j.get_m()) * dot_prod / distance_square;
            i.dvx -= scalar * delta_x * j.get_m();
            i.dvy -= scalar * delta_y * j.get_m();
            // now relax the distance a bit: after the collision, there must be
            // at least (ratio * radius) between them
            i.dx += delta_x / distance * ratio * radius / 2.0;
            i.dy += delta_y / distance * ratio * radius / 2.0;
        }
        else
        {
            // update acceleration only when no collision
            auto scalar = gravity / distance_square / distance;
            i.get_ax() -= scalar * delta_x * j.get_m();
            i.get_ay() -= scalar * delta_y * j.get_m();
        }
    }
};
