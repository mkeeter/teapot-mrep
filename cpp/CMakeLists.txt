cmake_minimum_required(VERSION 3.12)
project(mrep)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(mrep main.cc)
target_link_libraries(mrep Eigen3::Eigen)
set_target_properties(mrep PROPERTIES CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DMALLOC_CHECKS")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DEIGEN_NO_DEBUG -O3")
