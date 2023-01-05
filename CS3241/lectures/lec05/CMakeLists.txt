# How to compile:
#	  1. cmake -S ./ -B ./build -DFILE=<FILE_NAME>
#   2. cmake --build ./build

cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
	cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

# Set up the project
project(${FILE})

# Suppress generation of ZERO_CHECK build target
set(CMAKE_SUPPRESS_REGENERATION true)

# Set C++17 standard, no GNU extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add the executable
add_executable(${PROJECT_NAME} ${FILE}.cpp)

# Set the output directory to the project directory
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/$<0:>)

if(WIN32 AND MSVC)
	# Use provided libraries and headers if Windows and MSVC
	target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
	target_compile_definitions(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:NDEBUG> FREEGLUT_STATIC)
	target_link_libraries(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/lib/freeglut.lib)
elseif(APPLE)
	# When on MacOS, use system-installed GLUT and OpenGL
	find_package(GLUT REQUIRED)
	find_package(OpenGL REQUIRED)
	target_include_directories(${PROJECT_NAME} PRIVATE ${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
	target_link_libraries(${PROJECT_NAME} OpenGL::GL OpenGL::GLU GLUT::GLUT)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
	# On Linux, likewise, but set GLVND imported targets, use system-installed GLUT and OpenGL
	find_package(GLUT REQUIRED)
	find_package(OpenGL REQUIRED)
	set(OpenGL_GL_PREFERENCE GLVND)
	target_include_directories(${PROJECT_NAME} PRIVATE ${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
	target_link_libraries(${PROJECT_NAME} OpenGL::OpenGL OpenGL::GLU GLUT::GLUT)
endif()
