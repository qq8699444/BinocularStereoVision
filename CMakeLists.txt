cmake_minimum_required( VERSION 2.6 )
project( BinocularStereoVision )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -Wall")

set_property(GLOBAL PROPERTY USE_FOLDERS On)
include_directories( E:/opensrc/eigen-eigen-323c052e1731/)
include_directories( E:/opencv/opencv-410-build/install/include/)

set(OpenCV_DIR E:/opencv/opencv-410-build/install)
#include_directories( E:/opencv/opencv-410-build/install)
find_package(OpenCV REQUIRED)


add_executable(dltCalibration dltCalibration.cpp dltalgo.cpp parser.cpp)
target_link_libraries(dltCalibration ${OpenCV_LIBS} ) 

add_executable(tsaiCalibration tsaiCalibration.cpp)
target_link_libraries(tsaiCalibration ${OpenCV_LIBS} ) 

add_executable(zzhangCalibration zzhangCalibration.cpp dltalgo.cpp parser.cpp)
target_link_libraries(zzhangCalibration ${OpenCV_LIBS} ) 

add_executable(mrginghamChessBoardDetector mrginghamChessBoardDetector.cpp mrgingham.cpp chess.cpp find_chessboard_corners.cpp find_blobs.cpp find_grid.cpp)
target_link_libraries(mrginghamChessBoardDetector ${OpenCV_LIBS} ) 

add_executable(gaussNewton gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS} ) 


