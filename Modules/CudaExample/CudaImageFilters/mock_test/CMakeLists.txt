set(CPP_FILES
    TestDeviceUniquePtr_MUT.cpp
)

create_cuda_module_test(MitkCudaImageFilters_MUT
    DEPENDS MitkMockCudaRuntime
    CPP_FILES ${CPP_FILES}
)
