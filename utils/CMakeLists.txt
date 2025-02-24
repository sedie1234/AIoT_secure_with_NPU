cmake_minimum_required(VERSION 3.10)
project(utils)

set(UTILS_SOURCES
    gal251_field.c
    matrix.c
    model_utils.c
    onnx.pb-c.c
)

add_library(utils STATIC ${UTILS_SOURCES})

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(utils PRIVATE protobuf-c)