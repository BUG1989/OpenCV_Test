cmake_minimum_required(VERSION 2.8.10)

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

PROJECT(OpenCV_Test)

set(CMAKE_BUILD_TYPE release)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# support C++11
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# support C++14, when gcc version > 5.1, use -std=c++14 instead of c++1y
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

IF (BUILD_MODE STREQUAL "debug")
	MESSAGE(STATUS "****** BUILD MODE: DEBUG ******")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -O2")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -g -Wall -O2")
ELSE()
	MESSAGE(STATUS "====== BUILD MODE: REALSE ======")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
ENDIF()

MESSAGE(STATUS "project source dir: ${PROJECT_SOURCE_DIR}")
SET(PATH_TEST_FILES ${PROJECT_SOURCE_DIR}/./../../demo/OpenCV_Test)
SET(PATH_SRC_FILES ${PROJECT_SOURCE_DIR}/./../../src/fbc_cv)
MESSAGE(STATUS "path src files: ${PATH_TEST_FILES}")

# head file search path
INCLUDE_DIRECTORIES(
	${PATH_TEST_FILES}
	${PATH_SRC_FILES}/include
)

if(ANDROID)
    set(OpenCV_DIR /home/OpenCV-android-sdk/sdk/native/jni)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++11" )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -fPIE -pie")
elseif(ARM_LINUX)
    set(OpenCV_DIR /home/bug1989/opencv/hi3519/share/OpenCV)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIE -pie -std=c++11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
elseif(ARM_LINUX_64)
	set(OpenCV_DIR /home/bug1989/opencv/hi3559/share/OpenCV)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIE -pie -std=c++11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# find library
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs opencv_video opencv_videoio)
if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()

# recursive query match files :*.cpp, *c
FILE(GLOB_RECURSE TEST_CPP_LIST ${PATH_TEST_FILES}/*.cpp)
FILE(GLOB_RECURSE TEST_C_LIST ${PATH_TEST_FILES}/*.c)
#MESSAGE(STATUS "cpp list: ${TEST_CPP_LIST} ${TEST_C_LIST}")

FILE(GLOB_RECURSE SRC_CPP_LIST ${PATH_SRC_FILES}/*.cpp)
#MESSAGE(STATUS "src cpp list: ${SRC_CPP_LIST}")

# generate dynamic library for fbc_cv
ADD_LIBRARY(fbc_cv STATIC ${SRC_CPP_LIST})

# build executable program
ADD_EXECUTABLE(OpenCV_Test ${TEST_CPP_LIST} ${TEST_C_LIST})
# add dependent library: static and dynamic
# TARGET_LINK_LIBRARIES(OpenCV_Test fbc_cv ${opencv_core} ${opencv_imgproc} ${opencv_highgui} ${opencv_imgcodecs} ${opencv_video} ${opencv_videoio} pthread)
TARGET_LINK_LIBRARIES(OpenCV_Test fbc_cv ${OpenCV_LIBS})

install(TARGETS OpenCV_Test
				fbc_cv
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)  

