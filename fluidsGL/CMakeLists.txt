find_package(CUDA REQUIRED)
find_package(OpenGL 2.1 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})

# Target older architectures for those who haven't upgraded their graphics card to the latest models.
set(CUDA_NVCC_FLAGS "-gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 ")

cuda_add_library(fluidsGL_kernels
    defines.h
    fluidsGL_kernels.cu
)
cuda_add_cufft_to_target(fluidsGL_kernels)

# GLUT based
add_executable(fluidsGL
    helper_timer.h
    fluidsGL_kernels.h
    fluidsGL.cpp
)
target_link_libraries(fluidsGL fluidsGL_kernels ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})
set_target_properties(fluidsGL PROPERTIES COMPILE_FLAGS "-std=c++11")

# Qt based
find_package(Qt5Widgets)
if(Qt5Widgets_FOUND)
    set(CMAKE_AUTOMOC ON)
    find_package(Qt5OpenGL REQUIRED)
    include_directories(${Qt5Widgets_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS})
    add_executable(fluidsQt glfluids.cpp fluidsQt.cpp mainwindow.cpp)
    target_link_libraries(fluidsQt fluidsGL_kernels Qt5::Widgets Qt5::OpenGL ${OPENGL_LIBRARIES} )
    set_target_properties(fluidsQt PROPERTIES COMPILE_FLAGS "-std=c++11")
endif(Qt5Widgets_FOUND)
