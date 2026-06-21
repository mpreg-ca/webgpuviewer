cmake_minimum_required(VERSION 3.22.1)

project("resize")

add_library(resize SHARED
    resize.cpp
)

target_link_libraries(resize jnigraphics log)
