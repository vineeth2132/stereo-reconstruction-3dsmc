cmake_minimum_required(VERSION 3.20)

project(stereo_reconstruction
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

set(SOURCE_FILES
    src/main.cpp
    src/DataLoader.cpp
    src/SparseFeatureMatcher.cpp
    src/StereoGeometryEstimator.cpp
    src/StereoRectifier.cpp
    src/DenseStereoMatcher.cpp
    src/CustomDenseMatcher.cpp
    src/DepthReconstructor.cpp)

set(HEADER_FILES
    include/DataLoader.h
    include/SparseFeatureMatcher.h
    include/StereoGeometryEstimator.h
    include/StereoRectifier.h
    include/DenseStereoMatcher.h
    include/CustomDenseMatcher.h
    include/DepthReconstructor.h)

add_executable(stereo_reconstruction)

target_sources(stereo_reconstruction
    PRIVATE
        ${SOURCE_FILES}
        ${HEADER_FILES})

target_include_directories(stereo_reconstruction
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_compile_features(stereo_reconstruction PRIVATE cxx_std_20)

# The WLS disparity filter lives in the opencv_contrib ximgproc module, which is
# an optional vcpkg feature (opencv4[contrib]). Enable it only when present so the
# project still builds against a plain opencv4 install.
if(TARGET opencv_ximgproc)
    target_compile_definitions(stereo_reconstruction PRIVATE HAVE_OPENCV_XIMGPROC)
    message(STATUS "opencv_ximgproc found: WLS disparity filtering enabled.")
else()
    message(STATUS "opencv_ximgproc NOT found: building without WLS disparity filtering (install opencv4[contrib]).")
endif()

target_link_libraries(stereo_reconstruction
    PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
)

if(MSVC)
    target_compile_options(stereo_reconstruction
        PRIVATE
            /W4
            /permissive-
    )
else()
    target_compile_options(stereo_reconstruction
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()