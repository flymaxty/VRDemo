CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("D:\\Workspace\\windows_ws\\sdk\\Tesseract\\include")
LINK_DIRECTORIES("D:\\Workspace\\windows_ws\\sdk\\Tesseract\\lib")

ADD_EXECUTABLE(VRTest main.cpp)
if(${CMAKE_BUILD_TYPE} MATCHES "debug")
	TARGET_LINK_LIBRARIES(VRTest ${OpenCV_LIBS} libtesseract304d)
else()
	TARGET_LINK_LIBRARIES(VRTest ${OpenCV_LIBS} libtesseract304d)
endif()
