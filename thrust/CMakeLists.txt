find_package(CUDA REQUIRED)

cuda_add_executable(version version.cu)

cuda_add_executable(sort sort.cu)

cuda_add_executable(device-vector device-vector.cu)

cuda_add_executable(transformations transformations.cu)

cuda_add_executable(transform_reduce transform_reduce.cu)
