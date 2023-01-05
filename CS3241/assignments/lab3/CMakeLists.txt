cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
	cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()
# Set up the project
project(Lab3 VERSION 1.0
             DESCRIPTION "CS3241 Take-Home Lab Assignment 3: Reflective Table in a Room"
             LANGUAGES CXX)

# Suppress generation of ZERO_CHECK build target
set(CMAKE_SUPPRESS_REGENERATION true)

# Set C++17 standard, no GNU extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add the executable
add_executable(${PROJECT_NAME} main.cpp image_io.cpp)

# Set the output directory to the project directory
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/$<0:>)

if(WIN32 AND MSVC)
	# Use provided libraries and headers if Windows and MSVC
	target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/include)
	target_compile_definitions(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:NDEBUG> FREEGLUT_STATIC GLEW_STATIC)
	target_link_libraries(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/lib/freeglut_static.lib ${CMAKE_SOURCE_DIR}/lib/glew32s.lib)
	target_compile_options(${PROJECT_NAME} PRIVATE /W4)
elseif(APPLE)
	# When on MacOS, use system-installed GLUT and OpenGL
	find_package(GLUT REQUIRED)
	find_package(OpenGL REQUIRED)
	target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
		${OPENGL_INCLUDE_DIR}
		${GLUT_INCLUDE_DIR}
		${CMAKE_SOURCE_DIR}/include)
	target_link_libraries(${PROJECT_NAME} OpenGL::GL OpenGL::GLU GLUT::GLUT)
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
	# On Linux, likewise, but set GLVND imported targets, use system-installed GLUT, OpenGL, and GLEW
	find_package(GLUT REQUIRED)
	find_package(OpenGL REQUIRED)
	find_package(GLEW REQUIRED)
	set(OpenGL_GL_PREFERENCE GLVND)
	target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
		${OPENGL_INCLUDE_DIR}
		${GLUT_INCLUDE_DIR}
		${GLEW_INCLUDE_DIRS}
		${CMAKE_SOURCE_DIR}/include)
	target_link_libraries(${PROJECT_NAME} OpenGL::OpenGL OpenGL::GLU GLUT::GLUT GLEW::GLEW)
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)
endif()
