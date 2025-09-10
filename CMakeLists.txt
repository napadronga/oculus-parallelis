cmake_minimum_required(VERSION 3.10)
project(MyApp)

set(CMAKE_CXX_STANDARD 17)

# Tell CMake where to find OpenCV
set(OpenCV_DIR "C:/Projects/oculus-parallelis/opencv/build/install/x64/vc17/lib")

find_package(OpenCV REQUIRED)

add_executable(my_app main.cpp)

target_link_libraries(my_app PRIVATE ${OpenCV_LIBS})
