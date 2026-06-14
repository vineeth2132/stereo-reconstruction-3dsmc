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
    src/DepthReconstructor.cpp)

set(HEADER_FILES
    include/DataLoader.h
    include/SparseFeatureMatcher.h
    include/StereoGeometryEstimator.h
    include/StereoRectifier.h
    include/DenseStereoMatcher.h
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