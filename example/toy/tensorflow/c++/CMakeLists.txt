cmake_minimum_required(VERSION 3.5)
project(hello_tensorflow)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

link_directories(/usr/lib /usr/local/lib)
# Eigen
include_directories(/usr/local/include/eigen3)

# Protobuf
FIND_PACKAGE(Protobuf 3.4.0 REQUIRED)
MESSAGE(STATUS "Using protobuf ${Protobuf_VERSION}")

# Tensorflow
include_directories(/usr/local/include/google/tensorflow)

# Opencv
find_package(OpenCV REQUIRED)

add_executable(mnist mnist.cpp)
target_link_libraries(mnist tensorflow_cc ${Protobuf_LIBRARIES} ${OpenCV_LIBS})
