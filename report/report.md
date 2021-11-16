# CSC4005 Assignment 3
119010115 Yongjin, Huang

# Execution
## Build
```sh
cd /path/to/project
# 1. On Virtual Machine(without Cuda)
./debug.sh
# 2. On Slurm(with Cuda)
./release.sh
```
## Run
```sh
cd /path/to/project/build

```
# Introduction
An N-body simulation approximates the motion of particles, often specifically particles that interact with one another through some types of physical forces. Using this broad definition, the types of particles that can be simulated using n-body methods are quite significant, ranging from celestial bodies to individual atoms in a gas cloud. From here out, we will specialize the conversation to gravitational interactions, where individual particles are defined as a physical celestial body, such as a planet, star, or black hole. A motion of the particles on the bodies themselves is neglected since it is often not interesting to the problem and will hence add an unnecessarily large number of particles to the simulation. N-body simulation has numerous applications in areas such as astrophysics, molecular dynamics, and plasma physics. The simulation proceeds over time steps, each time computing the net force on each body and thereby updating its position and other attributes. If all pairwise forces are computed directly, this requires $O(N^2)$ operations at each time step.

In this assignment, the N-body simulation is in two dimensions. In order to visualize the N-body simulation, each body is modeled as a ball.
# Design
In each time step, the simulation is divided into two parts for each body:
1. Part One: Calculate the forces between it and the others, and check whether two bodies collide. Two bodies collide if their distance is less than the radius of the ball.
2. Part Two: Update its position, velocity based on the time elapse and check whether it crosses the boundary.

If there are $N$ bodies, the time complexity of the simulation in each time step is $O(N^2)$

The design of sequential version and parallel version differs a little.  
- In the sequential version, in Part One, if two bodies collide, their position and velocity are immediately updated. Thus, Part One computation is halved, and only $\frac{N(N-1)}{2}$ calculation is needed.
- In parallel versions, in Part One, if two bodies collide, the changes in their position and velocity are saved. Thus, their position and velocity are not updated in Part One but in Part Two. The reason for this design is to eliminate the data dependency between bodies.

This slight difference results in different movements of the bodies in visualization.

# Implementations
## Sequential Implementation
This sequential implementation is given by teaching assistants.  

```cpp
void update_for_tick(double elapse,
                    double gravity,
                    double position_range,
                    double radius)
{
    ax.assign(size(), 0);
    ay.assign(size(), 0);
    // Part One
    for (size_t i = 0; i < size(); ++i)
    {
        for (size_t j = i + 1; j < size(); ++j)
        {
            // update both bodies' positions and velocities immediately
            check_and_update(get_body(i), get_body(j), radius, gravity);
        }
    }
    // Part Two
    for (size_t i = 0; i < size(); ++i)
    {
        get_body(i).update_for_tick(elapse, position_range, radius);
    }
}
```

## MPI Implementation
| ![Figure 1](mpi-implementation.png) |
| :---------------------------------: |
|    Figure 1: MPI Implementation     |

The MPI implementation contains the following steps:
1. The root process creates a body pool and broadcast it to every other process.
2. Each process calculates its range of bodies. Bodies are distributed evenly across processes.
3. Each process runs Part One and Part Two.
4. Gather bodies data to the root process.

```cpp
void worker(int rank, int world_size)
{
    // Preparation: Broadcast all bodies to all processes
    // ...
    // Each process calculate its range of bodies
    pool.clear_acceleration();
    int elements_per_process = pool.size() / world_size;
    size_t st_idx = elements_per_process * rank;
    size_t end_idx = st_idx + elements_per_process;
    if (rank == world_size - 1)
        end_idx = pool.size();
    // Part One
    for (size_t i = st_idx; i < end_idx; i++)
    {
        for (size_t j = 0; j < pool.size(); j++)
        {
            if (i == j)
                continue;
            // update body i's position and velocity immediately
            pool.mpi_check_and_update(pool.get_body(i), pool.get_body(j), local_radius, local_gravity);
        }
    }
    // Part Two
    for (size_t i = st_idx; i < end_idx; i++)
    {
        pool.get_body(i).update_for_tick(local_elapse, local_space, local_radius);
    }
    // Gather bodies data to the root process
    // ...
}
```


## Pthread Implementation
| ![Figure 2](pthread-implementation.png) |
| :-------------------------------------: |
|    Figure 2: Pthread Implementation     |

The Pthread implementation contains the following steps:
1. The main thread distributes bodies range evenly and create child threads.
2. Each child thread gets its bodies range.
3. Each child thread runs Part One, and wait until all child threads complete Part One using a barrier. Then run Part Two.
```cpp
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

void *worker(void *data)
{
    struct idx_struct *p = reinterpret_cast<idx_struct *>(data);
    // Part One
    for (size_t i = p->st_idx; i < p->end_idx; i++)
    {
        for (size_t j = 0; j < pool.size(); ++j)
        {
            if (i == j)
                continue;
            // save the changes in body i's position and velocity
            pool.shared_memory_check_and_update(pool.get_body(i), pool.get_body(j), radius, gravity);
        }
    }
    pthread_barrier_wait(&barrier);
    // Part Two
    for (size_t i = p->st_idx; i < p->end_idx; i++)
    {
        // update body i's position and velocity using the saved changes in Part One
        pool.get_body(i).update_by_delta_vector();
        pool.get_body(i).update_for_tick(elapse, space, radius);
    }
    return nullptr;
}
```
## OpenMP Implementation
| ![Figure 3](openmp-implementation.png) |
| :------------------------------------: |
|    Figure 3: OpenMP Implementation     |

Similar to Pthread implementation, the OpenMP implementation contains the following steps:
1. Split Part One among working threads, and each working thread gets one or more bodies.
2. Each working thread runs Part One. After all working threads have finished Part One, they start to run Part Two.
```cpp
void schedule()
{
    pool.clear_acceleration();
    pool.init_delta_vector();
    // Part One
#pragma omp parallel for shared(pool)
    for (size_t i = 0; i < pool.size(); ++i)
    {
        for (size_t j = 0; j < pool.size(); ++j)
        {
            if (i == j)
                continue;
            // save the changes in body i's position and velocity
            pool.shared_memory_check_and_update(pool.get_body(i), pool.get_body(j), radius, gravity);
        }
    }
    // Part Two
#pragma omp parallel for shared(pool)
    for (size_t i = 0; i < pool.size(); ++i)
    {
        // update body i's position and velocity using the saved changes in Part One
        pool.get_body(i).update_by_delta_vector();
        pool.get_body(i).update_for_tick(elapse, space, radius);
    }
}
```
## CUDA Implementation
| ![Figure 4](cuda-implementation.png) |
| :----------------------------------: |
|    Figure 4: CUDA Implementation     |


Similar to Pthread implementation, the OpenMP implementation contains the following steps:
1. Create a block with many threads using CUDA.
1. Each thread gets its bodies range.
2. Each thread runs Part One, and wait until all child threads complete Part One. Then run Part Two.
```cpp
// worker() is called in this way:
int main()
{
    dim3 grid(1);
    dim3 block(thread_num);
    worker<<<grid, block>>>();
}

__global__ void worker()
{
    int thread_id = threadIdx.x;
    int elements_per_thread = pool->size / thread_num;
    int st_idx = elements_per_thread * thread_id;
    int end_idx = st_idx + elements_per_thread;
    if (thread_id == thread_num - 1)
        end_idx = pool->size;
    // Part One
    for (int i = st_idx; i < end_idx; i++)
    {
        for (int j = 0; j < pool->size; ++j)
        {
            if (i == j)
                continue;
            // save the changes in body i's position and velocity
            pool->shared_memory_check_and_update(pool->get_body(i), pool->get_body(j), radius, gravity);
        }
    }
    __syncthreads();
    // Part Two
    for (int i = st_idx; i < end_idx; i++)
    {
        // update body i's position and velocity using the saved changes in Part One
        pool->get_body(i).update_by_delta_var();
        pool->get_body(i).update_for_tick(elapse, space, radius);
    }
}
```
## Hybrid Implementation
|     ![Figure 5](hybrid-implementation.png)     |
| :--------------------------------------------: |
| Figure 5: Hybrid MPI and OpenMP Implementation |

```cpp
void worker(int rank, int world_size)
{
    // Preparation: Broadcast all bodies to all processes
    // ...
    // Part One
#pragma omp parallel for shared(pool)
    for (size_t i = st_idx; i < end_idx; i++)
    {
        for (size_t j = 0; j < pool.size(); j++)
        {
            if (i == j)
                continue;
            // update body i's position and velocity immediately
            pool.mpi_check_and_update(pool.get_body(i), pool.get_body(j), local_radius, local_gravity);
        }
    }
#pragma omp parallel for shared(pool)
    // Step 2
    for (size_t i = st_idx; i < end_idx; i++)
    {
        pool.get_body(i).update_for_tick(local_elapse, local_space, local_radius);
    }
    // Gather bodies data to the root process
    // ...

}
```
# Results
Three different problem sizes ranging from small $200$, medium $400$ to large $800$ are tested. The amount of points is $size^2$ since they are in a square.

The performance of the program is analyzed from the following three dimensions:
- Duration in nanoseconds(ns)
- Speedup $Speedup_n = \frac{T_1}{T_n}$ where $T_1$ is the execution time on one process and $T_n$ is the execution time on $n$ processes.
- Speed, which is measured by pixels calculated per second.
## Cyclic vs. Block
I use the [cyclic distribution method](https://slurm.schedmd.com/sbatch.html#OPT_cyclic) instead of the default block distribution method to reduce the influence of inter-node communication costs. 
> The cyclic distribution method will distribute tasks to a node such that consecutive tasks are distributed over consecutive nodes (in a round-robin fashion)

Since when I adopted the default block distribution method, some data show that there is a significant drop between core sizes $32$ and $33$. I guess the reason is that when $33$ cores are required, the default block distribution method will offer $32$ cores on the same node and $1$ core on the other node, and inter-node communication time is much longer than intra-node communication time. 

The following table shows the significant difference between core sizes $32$ and $33$
| world size | problem size | duration(ns) | speed(px/s) |  speedup  |
| :--------: | :----------: | :----------: | :---------: | :-------: |
|     32     |     200      |   1927916    |  103739.0   | 18.335724 |
|     33     |     200      |   10680568   |   18725.6   | 3.309724  |
|     32     |     400      |   5519552    |   72469.6   | 25.721739 |
|     33     |     400      |   15070290   |   26542.3   | 9.420686  |
|     32     |     800      |   19809013   |   40385.7   | 28.641585 |
|     33     |     800      |   27092379   |   29528.6   | 20.941739 |
## MPI

| ![Figure 3](mpi-cyclic-duration-overview.png) |
| :-------------------------------------------: |
|        Figure 3: MPI Duration Overview        |

In Figure 3, the large gap between the blue curve and the orange curve shows that the sequential version is much slower than the MPI version. As problem size increases, the gap becomes even larger.


| ![Figure 4](mpi-cyclic-speedup-overview.png) |
| :------------------------------------------: |
|        Figure 4: MPI Speedup Overview        |

In Figure 4, for those process numbers greater than $1$, as problem size increases, the speedup also increases. The larger the process number, the faster the growth (bigger slope). 
| ![Figure 5](mpi-cyclic-speed-overview.png) |
| :----------------------------------------: |
|        Figure 5: MPI Speed Overview        |

In Figure 5, it can be noticed that greater process number brings faster speed. As the problem size increases, the speed decreases for all process numbers.

|    ![Figure 6](mpi-cyclic-speedup-200.png)    |
| :-------------------------------------------: |
| Figure 6: MPI Speedup with Problem Size $200$ |

|    ![Figure 7](mpi-cyclic-duration-200.png)    |
| :--------------------------------------------: |
| Figure 7: MPI Duration with Problem Size $200$ |

In Figure 6, initially, the speedup first increases with the process number. When the process number reaches about $30$, the speedup starts to fluctuate up and down in a slow downward trend. According to Figure 7, when the process number is above $20$, the duration is only about $2.5ms$. The possible reason is that the computation time of each process is small, and the inter-process communication time becomes the determining factor.

|    ![Figure 8](mpi-cyclic-speedup-400.png)    |
| :-------------------------------------------: |
| Figure 8: MPI Speedup with Problem Size $400$ |

In Figure 8, initially, the speedup first increases with the process number. When the process number reaches about $50$, the speedup starts to fluctuate up and down in a slow downward trend. I think the reason is the same as above.

|    ![Figure 9](mpi-cyclic-speedup-800.png)    |
| :-------------------------------------------: |
| Figure 9: MPI Speedup with Problem Size $800$ |

In Figure 9, the speedup keeps increasing with the process number. Since the computation workload is much larger than the above two situations, the speedup does not go down.
## Pthread and Comparing MPI and Pthread
For Pthread, the thread number increases from $1$ to $32$, each time by $1$, since Pthread requires shared memory and on a single node, there are only $32$ physical threads available.
| ![Figure 10](pthread-duration-overview.png) | ![Figure 11](mpi-duration-overview-compare.png) |
| :-----------------------------------------: | :---------------------------------------------: |
|    Figure 10: Pthread Duration Overview     |   Figure 11: MPI Duration Overview Comparison   |

In Figure 10, the large gap between the blue curve and the orange curve shows that the sequential version is much more slower than the Pthread version. As problem size increases, the gap becomes even larger.  
Compared to MPI version, pthread version shares similar results.
| ![Figure 12](pthread-speedup-overview.png) | ![Figure 13](mpi-speedup-overview-compare.png) |
| :----------------------------------------: | :--------------------------------------------: |
|    Figure 12: Pthread Speedup Overview     |   Figure 13: MPI Speedup Overview Comparison   |

In Figure 12, for those thread number greater than $1$, as problem size increases, the speedup also increases. The larger the thread number, the faster the growth (bigger slope).   
Compared to MPI version, the slope of the curve and the speedup is much smaller.
| ![Figure 14](pthread-speed-overview.png) | ![Figure 15](mpi-speed-overview-compare.png) |
| :--------------------------------------: | :------------------------------------------: |
|    Figure 14: Pthread Speed Overview     |   Figure 15: MPI Speed Overview Comparison   |

In Figure 14, it can be noticed that greater thread number brings faster speed. As the problem size increases, the speed decreases for all thread numbers.  
Compared to the MPI version, the speed is much smaller.

# Conclusion
By working on this assignment, I gain a deeper and better understanding of parallel programming. Not only the algorithm should be considered, but also the underlying hardware structure. Compared to MPI, Pthread is easier to implement, and it requires a shared memory hardware architecture. I compared different kinds of Load balancing algorithms and decided to choose the Master-Worker Scheme. 
# References
- https://openmp.org/wp-content/uploads/HybridPP_Slides.pdf
