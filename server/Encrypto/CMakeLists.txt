cmake_minimum_required(VERSION 3.10)
project(Encrypto)

add_subdirectory(../../utils utils_build)


################## enc model  #################
set(SOURCES src/encrypt.c
            src/main.c)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

add_executable(encrypto ${SOURCES})

target_link_libraries(encrypto PRIVATE 
                    utils
                    ${PROTOBUF_C_LIBRARIES})

target_include_directories(encrypto PRIVATE ../../utils)



################## enc data ###################
set(ENC_DATA_SOURCES src/main2.c)

add_executable(encrypt_data ${ENC_DATA_SOURCES})
target_link_libraries(encrypt_data PRIVATE 
                    utils
                    ${PROTOBUF_C_LIBRARIES})

target_include_directories(encrypt_data PRIVATE ../../utils)

################## enc file ###################
set(ENC_FILE_SOURCES src/main3.c)

add_executable(encrypt_file ${ENC_FILE_SOURCES})
target_link_libraries(encrypt_file PRIVATE 
                    utils
                    ${PROTOBUF_C_LIBRARIES})

target_include_directories(encrypt_file PRIVATE ../../utils)

