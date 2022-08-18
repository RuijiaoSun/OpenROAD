#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 3.9)

#Name your project here
project(small)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

#default to release mod
if(NOT CMAKE_BUILD_TYPE) 
	set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

enable_language(CUDA)
find_package(CUDA REQUIRED)

add_executable(small
			main.cpp
			solvers.cu
)

target_link_libraries(small
			${CUDA_cusparse_LIBRARY}
			${CUDA_cusolver_LIBRARY}
)