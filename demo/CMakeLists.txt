cmake_minimum_required(VERSION 2.8)

project( Demo )

find_package( OpenCV REQUIRED )
set(OpenCV_INCLUDE_DIRS "/home/ya_streltsova/build/OV_ARM_package/opencv/include")
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(InferenceEngine)
set(InferenceEngine_INCLUDE_DIRS "/home/ya_streltsova/build/OV_ARM_package/deployment_tools/inference_engine/include")
include_directories(${InferenceEngine_INCLUDE_DIRS})

add_executable( Demo src.cpp )
target_link_libraries( Demo ${OpenCV_LIBS}  ${InferenceEngine_LIBRARIES})