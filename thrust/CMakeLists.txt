find_package(CUDA REQUIRED)

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
endif()

cuda_add_executable(version version.cu)

cuda_add_executable(device-vector device-vector.cu)

cuda_add_executable(transformations transformations.cu)

cuda_add_executable(transform_reduce transform_reduce.cu)

cuda_add_executable(sum sum.cu)

cuda_add_executable(sort sort.cu)

cuda_add_executable(random random.cu)
