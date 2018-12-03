cmake_minimum_required(VERSION 2.8)

add_executable(face_detect src/main.cpp src/faceArray.cpp)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	include_directories(${OpenCV_INCLUDE_DIRS})
	target_link_libraries(face_detect ${OpenCV_LIBRARIES})
else(OpenCV_FOUND)
	message(FATAL ERROR "Librarie OpenCV introuvable!")
endif(OpenCV_FOUND)

