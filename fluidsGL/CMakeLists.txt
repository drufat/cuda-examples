find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})

cuda_add_library(fluidsGL_kernels
    fluidsGL_kernels.h
    fluidsGL_kernels.cuh
    fluidsGL_kernels.cu
)
cuda_add_cufft_to_target(fluidsGL_kernels)

add_executable(fluidsGL
    defines.h
    fluidsGL.cpp)
target_link_libraries(fluidsGL fluidsGL_kernels ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
