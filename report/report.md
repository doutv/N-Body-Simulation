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
# Introduction
An N-body simulation approximates the motion of particles, often specifically particles that interact with one another through some type of physical forces. Using this broad definition, the types of particles that can be simulated using n-body methods are quite significant, ranging from celestial bodies to individual atoms in a gas cloud. From here out, we will specialize the conversation to gravitational interactions, where individual particles are defined as a physical celestial body, such as a planet, star, or black hole. Motion of the particles on the bodies themselves is neglected, since it is often not interesting to the problem and will hence add an unnecessarily large number of particles to the simulation. N-body simulation have numerous applications in areas such as astrophysics, molecular dynamics and plasma physics. The simulation proceeds over time steps, each time computing the net force on every body and thereby updating its position and other attributes. If all pairwise ofrces are computed directly, this requires $O(N^2)$ opertaions at each time step.

In order to visualize the N-body simulation, each body is modeled as a ball
# Design

# Implementations
## Sequential Implementation
This sequential implementation is given by teaching assistants with small modifications, and here I just explain some variables:
- `buffer` is a `struct` wrapping a 1-D vector.
- The total number of points (pixels) is given by $size^2$, and the generated image is a square with side length $size$.
- `k_value` specifies the number of iterations for each point.
- In order to visualize the Mandelbrot Set, the point position in the complex plane should be transformed to position in the square image. Thus, variables `scale`, `x_center`, `y_center` are used.
```cpp
void sequential_calculate(Square &buffer, int size, int scale, double x_center, double y_center, int k_value, int world_size)
{
    double cx = static_cast<double>(size) / 2 + x_center;
    double cy = static_cast<double>(size) / 2 + y_center;
    double zoom_factor = static_cast<double>(size) / 4 * scale;
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            double x = (static_cast<double>(j) - cx) / zoom_factor;
            double y = (static_cast<double>(i) - cy) / zoom_factor;
            std::complex<double> z{0, 0};
            std::complex<double> c{x, y};
            int k = 0;
            do
            {
                z = z * z + c;
                k++;
            } while (norm(z) < 2.0 && k < k_value);
            buffer[{i, j}] = k;
        }
    }
}
```

## MPI Implementation
| ![Figure 1](mpi-implementation.png) |
| :---------------------------------: |
|    Figure 1: MPI Implementation     |

Master-Worker Scheme algorithm is implemented using MPI:
1. The master node is the node with rank $0$, and the worker nodes are the others. Assume there are $n-1$ worker nodes.
2. The points are divided into $size$ tasks, and each task is dynamically assigned to a worker node.

Master node `root_schedule`:
1. First, the master node broadcasts several variables to all worker nodes.
2. Then, it starts an iteration for $size+n-1$ rounds, in each round:
 1. It tries to receive data from any worker node. 
 2. After it receives data from a worker node, it sends the next task to this worker node. 
     1. In the first $size$ rounds, it sends a y-axis index to the worker node telling the worker node to compute all points with this y-axis index. 
     2. In the last $n-1$ rounds, it sends an invalid y-axis index to the worker node telling the worker node to stop calculation.
 3. If the received data is part of the results, store the received data.

```cpp
void root_schedule(Square &result, size_t size, int scale, double x_center, double y_center, int k_value, size_t world_size)
{
    MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&scale, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&x_center, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&y_center, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> recv_buffer(size + 1, 0);
    for (size_t y_idx = 0; y_idx < size + world_size - 1; y_idx++)
    {
        MPI_Status status;
        MPI_Recv(recv_buffer.data(), size + 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        MPI_Send(&y_idx, 1, MPI_UNSIGNED_LONG, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
        size_t recv_y_idx = recv_buffer.back();
        if (recv_y_idx < size)
            std::copy(recv_buffer.begin(), recv_buffer.begin() + size, result.buffer.begin() + recv_y_idx * size);
    }
}
```

Worker node `slave_calculate`:
1. First, the worker node receives several variables from the master node.
2. Then, it creates an empty `vector` to temporally store results and send this to the master node to tell the master node it is idle.
3. Finally, it starts an infinite loop:
   1. First, it tries to receive a y-axis index from the master node. 
   2. If this index is invalid, it will stop the process.
   3. If this index is valid, it will do the calculation similar to the sequential version.
   4. After calculation, it sends the results back to the master node.

```cpp
void slave_calculate()
{
    size_t size;
    int scale, k_value;
    double x_center, y_center;
    MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&scale, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&x_center, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&y_center, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double cx = static_cast<double>(size) / 2 + x_center;
    double cy = static_cast<double>(size) / 2 + y_center;
    double zoom_factor = static_cast<double>(size) / 4 * scale;
    std::vector<int> shard_result(size + 1, INT32_MAX);
    MPI_Send(shard_result.data(), size + 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    while (1)
    {
        size_t y_idx;
        MPI_Recv(&y_idx, 1, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (y_idx >= size)
            break;
        // calculation
        double x = (static_cast<double>(y_idx) - cx) / zoom_factor;
        for (size_t x_idx = 0; x_idx < size; ++x_idx)
        {
            double y = (static_cast<double>(x_idx) - cy) / zoom_factor;
            std::complex<double> z{0, 0};
            std::complex<double> c{x, y};
            int k = 0;
            do
            {
                z = z * z + c;
                k++;
            } while (norm(z) < 2.0 && k < k_value);
            shard_result[x_idx] = k;
        }
        shard_result.back() = y_idx;
        MPI_Send(shard_result.data(), size + 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}
```

## Pthread Implementation
| ![Figure 2](pthread-implementation.png) |
| :-------------------------------------: |
|    Figure 2: Pthread Implementation     |

A modified version of the Master-Worker Scheme algorithm is implemented using Pthread:
1. Instead of using a master node to schedule tasks, each node uses a global mutex lock to get its task from the global shared memory.
2. The Mandelbrot Set is divided into $size$ tasks using the Round-robin method. Thus, each task contains almost the same amount of points.
3. If a node is idle, it tries to acquire the global mutex lock and get its next task:
   1. By calculating the y-axis index range and modifying the next y-axis index to calculate, a node is assigned a task.
   2. After calculation, a node stores the results in the global result memory space. Since nodes should not have overlapping y-axis index, writing to the global result memory space concurrently without the protection of mutex lock should be fine.

```cpp
// global constants
Square canvas(100);
int thread_num;
int size;
int scale;
int x_center;
int y_center;
int k_value;

// global shared memory
int element_per_thread;
int remain;
int g_idx;
int cur_task;
int num_of_tasks;

std::mutex g_mutex;

void *thread_calculate(void *)
{
    while (1)
    {
        size_t st_idx, end_idx;
        if (cur_task >= num_of_tasks)
            break;
        g_mutex.lock();
        // -----critical section start-----
        end_idx = cur_task < remain ? g_idx + element_per_thread + 1 : g_idx + element_per_thread;
        ++cur_task;
        st_idx = g_idx;
        g_idx = end_idx;
        // -----critical section end-----
        g_mutex.unlock();
        for (size_t y_idx = st_idx; y_idx < end_idx; y_idx++)
        {
            double cx = static_cast<double>(size) / 2 + x_center;
            double cy = static_cast<double>(size) / 2 + y_center;
            double zoom_factor = static_cast<double>(size) / 4 * scale;
            double x = (static_cast<double>(y_idx) - cx) / zoom_factor;
            for (size_t x_idx = 0; x_idx < static_cast<size_t>(size); ++x_idx)
            {
                double y = (static_cast<double>(x_idx) - cy) / zoom_factor;
                std::complex<double> z{0, 0};
                std::complex<double> c{x, y};
                int k = 0;
                do
                {
                    z = z * z + c;
                    k++;
                } while (norm(z) < 2.0 && k < k_value);
                canvas[{x_idx, y_idx}] = k;
            }
        }
    }
    return nullptr;
}

void schedule()
{
    canvas.resize(size);
    std::vector<pthread_t> threads(thread_num);
    num_of_tasks = size;
    element_per_thread = size / num_of_tasks;
    remain = size % num_of_tasks;
    g_idx = 0;
    cur_task = 0;
    for (auto &thread : threads)
    {
        pthread_create(&thread, nullptr, thread_calculate, nullptr);
    }
    for (auto &thread : threads)
    {
        pthread_join(thread, nullptr);
    }
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