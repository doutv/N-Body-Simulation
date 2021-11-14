cmake_minimum_required(VERSION 3.2)
project(a3 CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "-G -g")
endif()

# option(CUDA_ONLY "Build Cuda Only" OFF)

if (WIN32)
    include_directories(include)
    add_executable(cuda src/cuda.cu)
    set_target_properties(cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    find_package(SDL2 REQUIRED)
    find_package(Freetype REQUIRED)
    find_package(MPI REQUIRED)
    find_package(Threads REQUIRED)
    set(OpenGL_GL_PREFERENCE "GLVND")
    find_package(OpenGL REQUIRED)
    find_package(OpenMP REQUIRED)

    include_directories(
            include
            imgui
            imgui/backends
            ${SDL2_INCLUDE_DIRS}
            ${FREETYPE_INCLUDE_DIRS}
            ${MPI_CXX_INCLUDE_DIRS})

    file(GLOB IMGUI_SRC
            imgui/*.cpp
            imgui/backends/imgui_impl_sdl.cpp
            imgui/backends/imgui_impl_opengl2.cpp
            imgui/misc/freetype/imgui_freetype.cpp
            imgui/misc/cpp/imgui_stdlib.cpp
            )
    add_library(core STATIC ${IMGUI_SRC})
    get_filename_component(FONT_PATH imgui/misc/fonts/DroidSans.ttf ABSOLUTE)
    target_link_libraries(core PUBLIC
            Freetype::Freetype SDL2::SDL2 OpenGL::GL ${CMAKE_DL_LIBS} 
            Threads::Threads ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
    target_compile_definitions(core PUBLIC -DImDrawIdx=unsigned)
    target_compile_options(core PUBLIC -Werror -Wall -Wextra -Wpedantic)
    target_compile_definitions(core PUBLIC -DFONT_PATH=\"${FONT_PATH}\")

    add_executable(template src/graphic.cpp src/template.cpp)
    target_link_libraries(template core)

    add_executable(pthread_imgui src/graphic.cpp src/pthread_imgui.cpp)
    target_link_libraries(pthread_imgui core)

    add_executable(mpi_imgui src/graphic.cpp src/mpi_imgui.cpp)
    target_link_libraries(mpi_imgui core)

    add_executable(openmp_imgui src/graphic.cpp src/openmp_imgui.cpp)
    target_link_libraries(openmp_imgui core)
    target_link_libraries(openmp_imgui ${OpenMP_CXX_LIBRARIES})
    set(CMAKE_CXX_FLAGS "-fsanitize=thread")

    add_executable(cuda_imgui src/graphic.cpp src/cuda_imgui.cu)
    target_link_libraries(cuda_imgui core)
    set_target_properties(cuda_imgui PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
