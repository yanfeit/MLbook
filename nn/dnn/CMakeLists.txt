cmake_minimum_required(VERSION 3.0)

PROJECT(dnn)

# .. -> hint, that the mnist package is one directory level above.
# When using just "find_package(MNIST REQUIRED)", "MNIST_DIR"
#    cmake variable has to be set correctly.
find_package(MNIST PATHS ..)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
  endif(NOT MNIST_FOUND)

  if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${MNIST_INCLUDE_DIR})
add_executable(dnn main.cpp)
target_compile_features(dnn PRIVATE cxx_range_for)

# Pass MNIST data directory to main.cpp
target_compile_definitions(dnn PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
