cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
	cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

# Set up the project
project(Lab4 VERSION 1.0
             DESCRIPTION "CS3241 Take-Home Assignment 4: Ray-Tracing"
             LANGUAGES CXX)

# Suppress generation of ZERO_CHECK build target
set(CMAKE_SUPPRESS_REGENERATION true)

# Set C++ standard to C++17, no extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add the executable
add_executable(${PROJECT_NAME} Main.cpp Camera.cpp Image.cpp ImageIO.cpp Raytrace.cpp Util.cpp Plane.cpp Sphere.cpp Triangle.cpp)

# Set the output directory to the top-level directory of the project
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/$<0:>)

# Include the stb_image directory
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

# If not Debug config, then find and link OpenMP for Windows and Linux targets using generator expression
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(WIN32 OR CMAKE_SYSTEM_NAME MATCHES "Linux")
        find_package(OpenMP REQUIRED)
        target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
