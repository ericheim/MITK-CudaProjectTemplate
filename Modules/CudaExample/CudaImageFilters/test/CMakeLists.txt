
set(H_FILES
    TestCudaMinMaxScalerFixture.h
)

set(CPP_FILES
    TestCudaMinMaxScaler.cpp
    TestDeviceUniquePtr.cpp
)

create_cuda_module_test(MitkCudaImageFilters
    WITH_CUDA
    CPP_FILES ${CPP_FILES}
    H_FILES ${H_FILES}
)
