cmake_minimum_required( VERSION 2.6 )
project( BinocularStereoVision )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -Wall")

set_property(GLOBAL PROPERTY USE_FOLDERS On)
include_directories( E:/opensrc/eigen-eigen-323c052e1731/)
include_directories( E:/opencv/opencv-410-build/install/include/)

set(OpenCV_DIR E:/opencv/opencv-410-build/install)
#include_directories( E:/opencv/opencv-410-build/install)
find_package(OpenCV REQUIRED)


add_executable(dlt dlt.cpp)
target_link_libraries(dlt ${OpenCV_LIBS} ) 

