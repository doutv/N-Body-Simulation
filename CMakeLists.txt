cmake_minimum_required(VERSION 3.2)
project(a3 CXX)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode: add race condition detector")
    set(CMAKE_CXX_FLAGS "-fsanitize=thread")
endif()

find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)

include_directories(
        include
        imgui
        imgui/backends
        ${SDL2_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_DIRS}
        )

file(GLOB IMGUI_SRC
        imgui/*.cpp
        imgui/backends/imgui_impl_sdl.cpp
        imgui/backends/imgui_impl_opengl2.cpp
        imgui/misc/freetype/imgui_freetype.cpp
        imgui/misc/cpp/imgui_stdlib.cpp
        )
add_library(core STATIC ${IMGUI_SRC} src/pthread_barrier.c)
get_filename_component(FONT_PATH imgui/misc/fonts/DroidSans.ttf ABSOLUTE)
target_link_libraries(core PUBLIC
    Freetype::Freetype SDL2::SDL2 
    OpenGL::GL ${CMAKE_DL_LIBS} 
    Threads::Threads ${MPI_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX ${OpenMP_CXX_LIBRARIES}
    )
target_compile_definitions(core PUBLIC -DImDrawIdx=unsigned)
target_compile_options(core PUBLIC)
target_compile_definitions(core PUBLIC -DFONT_PATH=\"${FONT_PATH}\")

add_executable(pthread_imgui src/graphic.cpp src/pthread_imgui.cpp src/pthread_barrier.c)
target_link_libraries(pthread_imgui core)
target_compile_options(pthread_imgui PRIVATE -Werror -Wall -Wextra -Wpedantic)
add_executable(pthread src/pthread.cpp src/pthread_barrier.c)
target_link_libraries(pthread core)
target_compile_options(pthread PRIVATE -Werror -Wall -Wextra -Wpedantic)

add_executable(mpi_imgui src/graphic.cpp src/mpi_imgui.cpp)
target_link_libraries(mpi_imgui core)
target_compile_options(mpi_imgui PRIVATE -Werror -Wall -Wextra -Wpedantic)
add_executable(mpi src/mpi.cpp)
target_link_libraries(mpi core)
target_compile_options(mpi PRIVATE -Werror -Wall -Wextra -Wpedantic)

add_executable(openmp_imgui src/graphic.cpp src/openmp_imgui.cpp)
target_link_libraries(openmp_imgui core)
target_compile_options(openmp_imgui PRIVATE -Werror -Wall -Wextra -Wpedantic)
add_executable(openmp src/openmp.cpp)
target_link_libraries(openmp core)
target_compile_options(openmp PRIVATE -Werror -Wall -Wextra -Wpedantic)

add_executable(hybrid_imgui src/graphic.cpp src/hybrid_mpi_openmp_imgui.cpp)
target_link_libraries(hybrid_imgui core)
target_compile_options(hybrid_imgui PRIVATE -Werror -Wall -Wextra -Wpedantic)
add_executable(hybrid src/hybrid_mpi_openmp.cpp)
target_link_libraries(hybrid core)
target_compile_options(hybrid PRIVATE -Werror -Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("release mode include cuda")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/opt/rh/devtoolset-10/root/usr/bin/gcc -gencode=arch=compute_75,code=[sm_75,compute_75]")
    add_executable(cuda_imgui src/graphic.cpp src/cuda_imgui.cu)
    target_link_libraries(cuda_imgui core)
    set_target_properties(
        cuda_imgui
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES OFF
        )

    add_executable(cuda src/cuda.cu)
    set_target_properties(
        cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES OFF
        )
endif()