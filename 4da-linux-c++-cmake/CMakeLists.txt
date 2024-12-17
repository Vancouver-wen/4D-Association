cmake_minimum_required(VERSION 3.0)
project(BaseProject)

set(CMAKE_BUILD_TYPE "Release") # Release Debug 默认 Debug

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

find_package( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(cli_demo)
add_subdirectory(four_d_association)
