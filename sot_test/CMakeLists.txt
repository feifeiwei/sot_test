cmake_minimum_required(VERSION 2.8)

project(bsnn-apps)

set(CMAKE_BUILD_TYPE Release)

#IF (CMAKE_BUILD_TYPE STREQUAL Debug)      
#    ADD_DEFINITIONS(-DDEBUG)           
#ENDIF()

find_package(OpenMP REQUIRED)

set(COREIP_SAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lstdc++fs -pthread -O3 -s -DNDEBUG ${OpenMP_CXX_FLAGS}")

# cmake 路径下的
#set(OpenCV_DIR /home/opencv_install/lib/cmake/opencv4)

# mf add
#link_directories(/home/official_code/lib)

#find_package(OpenCV 4.0.0 EXACT REQUIRED)
find_package(OpenCV EXACT REQUIRED)
find_package(Eigen3 REQUIRED) # for tracker

message(STATUS "OpenCV libaray information:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    header file: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "Eigen3 libaray information:")
message(STATUS "    version: ${Eigen3_VERSION}")
message(STATUS "    header: ${Eigen3_INCLUDE_DIRS}")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}       #  for bsnn_sot tracker

    )



find_library(ISP_LIB
            NAMES bstisp)
            #HINTS ${CMAKE_SOURCE_DIR}/3rdparty/bst_isp/lib
            #NO_CMAKE_FIND_ROOT_PATH)

if (NOT ISP_LIB)
    message(FATAL_ERROR "libbstisp.so Not Found!")
else()
    message (STATUS "ISP_LIB library found in ${ISP_LIB}")
endif()




find_library(
            BSNN_LIB
            NAMES bsnn
            HINTS ${CMAKE_SOURCE_DIR}/3rdparty/lib
            # CMAKE_FIND_ROOT_PATH is set to ONLY which limits the search path of find_library
            # refer to /opt/bstos/a1000/2.1.1.4/sysroots/x86_64-bstsdk-linux/usr/share/cmake/OEToolchainConfig.cmake
            # avoid it to make search path in HINTS work
            NO_CMAKE_FIND_ROOT_PATH
)
message("BSNN_LIB: ${BSNN_LIB}") 

if (NOT BSNN_LIB)
    message(FATAL_ERROR "libbsnn.so Not Found!")
endif (NOT BSNN_LIB)




add_executable(run_demo main.cpp
                        src/bsnn_sot_tracker.cpp
                        src/easylogging++.cc
                    )

target_link_libraries(run_demo  Eigen3::Eigen bstisp ${BSNN_LIB} ${ISP_LIB} ${OpenCV_LIBS} pthread stdc++fs  ${Eigen3_LIBRARY} OpenMP::OpenMP_CXX)