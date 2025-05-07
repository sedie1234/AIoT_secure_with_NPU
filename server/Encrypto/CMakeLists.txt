cmake_minimum_required(VERSION 3.10)
project(Encrypto)

add_subdirectory(../../utils utils_build)
################## enc data ###################
set(ENC_DATA_SOURCES src/main2.c)

add_executable(encrypt_data ${ENC_DATA_SOURCES})
target_link_libraries(encrypt_data PRIVATE 
                    utils)

target_include_directories(encrypt_data PRIVATE ../../utils)

################## enc file ###################
set(ENC_FILE_SOURCES src/main3.c)

add_executable(encrypt_file ${ENC_FILE_SOURCES})
target_link_libraries(encrypt_file PRIVATE 
                    utils)

target_include_directories(encrypt_file PRIVATE ../../utils)

