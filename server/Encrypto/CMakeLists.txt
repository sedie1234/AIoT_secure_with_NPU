cmake_minimum_required(VERSION 3.10)
project(Encrypto)

add_subdirectory(../../utils utils_build)

set(SOURCES src/encrypt.c
            src/main.c)

add_executable(encrypto ${SOURCES})

target_link_libraries(encrypto PRIVATE 
                    utils
                    ${PROTOBUF_C_LIBRARIES})

target_include_directories(encrypto PRIVATE ../../utils)