cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project(cpp)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cpp main.cpp
        lib/ConvNet.hpp
        lib/ConvNetClassifier.hpp
        lib/Dataset.hpp
        lib/utils.hpp)

target_link_libraries(cpp ${OpenCV_LIBS})
target_link_libraries(cpp "${TORCH_LIBRARIES}")
set_property(TARGET cpp PROPERTY CXX_STANDARD 17)