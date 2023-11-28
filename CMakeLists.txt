cmake_minimum_required(VERSION 3.1)
project(CloudMerger)

set(CMAKE_CXX_FLAGS "-std=c++14")

set(CMAKE_BUILD_TYPE release)

find_package(PCL REQUIRED)

add_executable(merge src/merge.cpp) 
target_link_libraries(merge ${PCL_LIBRARIES})
