cmake_minimum_required(VERSION 3.10)
project(Decrypto)

add_subdirectory(../../utils utils_build)

set(SOURCES src/decrypt.c
            src/main.c)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

#add_executable(decrypto ${SOURCES})

#target_link_libraries(decrypto PRIVATE 
#                    utils
#                    ${PROTOBUF_C_LIBRARIES})

#target_include_directories(decrypto PRIVATE ../../utils)

############# data decrypt #########
set(DATA_DECRYPT_SOURCES src/main2.c)

add_executable(decrypt_data ${DATA_DECRYPT_SOURCES})

target_link_libraries(decrypt_data PRIVATE 
                    utils
                    ${PROTOBUF_C_LIBRARIES})

target_include_directories(decrypt_data PRIVATE ../../utils)

############# file decrypt #########
set(DATA_DECRYPT_SOURCES src/main3.c)

add_executable(decrypt_file_cpu ${DATA_DECRYPT_SOURCES})

target_link_libraries(decrypt_file_cpu PRIVATE 
                    utils
                    ${PROTOBUF_C_LIBRARIES})

target_include_directories(decrypt_file_cpu PRIVATE ../../utils)

############# file decrypt #########
set(DATA_DECRYPT_SOURCES src/main4.c)

add_executable(decrypt_file_npu ${DATA_DECRYPT_SOURCES})

target_link_libraries(decrypt_file_npu PRIVATE 
                    utils
                    ${PROTOBUF_C_LIBRARIES})

target_include_directories(decrypt_file_npu PRIVATE ../../utils)
