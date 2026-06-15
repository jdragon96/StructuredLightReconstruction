cmake_minimum_required(VERSION 3.16)
project(StructuredLight LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

# ── Core library ────────────────────────────────────────────────────────
file(GLOB_RECURSE STRUCTURED_LIGHT_CORE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
)

add_library(StructuredLightCore STATIC
    ${STRUCTURED_LIGHT_CORE_SOURCES}
)
target_include_directories(StructuredLightCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(StructuredLightCore PUBLIC Eigen3::Eigen)

# ── Example executable ─────────────────────────────────────────────────
add_executable(MultiFrequencyTest
    example/MultiFrequencyTest.cpp
)
target_link_libraries(MultiFrequencyTest PRIVATE
    StructuredLightCore
    ${OpenCV_LIBS}
)
target_include_directories(MultiFrequencyTest PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(ColorCodedTest
    example/ColorCodedTest.cpp
)
target_link_libraries(ColorCodedTest PRIVATE
    StructuredLightCore
    ${OpenCV_LIBS}
)
target_include_directories(ColorCodedTest PRIVATE ${OpenCV_INCLUDE_DIRS})
