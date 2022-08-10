cmake_minimum_required(VERSION 3.21)
project(untitled)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
#MATH(EXPR stack_size "5120*1024*1024")
#set(CMAKE_EXE_LINKER_FLAGS "-Wl,--stack,${stack_size}")

add_executable(untitled main.cpp)
add_executable(untitled2 train.cpp)
add_executable(untitled3 test.cpp)
add_executable(untitled4 debug.cpp)