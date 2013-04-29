find_package(CUDA REQUIRED)

cuda_add_executable(hello-cuda hello-cuda.cu)

cuda_add_executable(cuda-test cuda-test.cu)

cuda_add_executable(fft fft.cu)

cuda_add_cufft_to_target(fft)

