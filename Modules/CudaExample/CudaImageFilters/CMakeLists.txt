
set(CPP_FILES
    CudaImageFilters.cpp
    CudaMinMaxScaler.cpp
    MinMaxScaler.cpp
)

set(CU_FILES
    CudaMinMaxScalerKernel.cu
)

set(CUH_FILES
    include/CudaMinMaxScalerStub.h
)

set(H_FILES
    include/CudaImageFilters.h
    include/CudaMinMaxScaler.h
    include/CudaMinMaxScalerStub.h
    include/ICudaMinMaxScalerStub.h
    include/DeviceUniquePtr.h
    include/MinMaxScaler.h
)

# main module with cuda image filters
mitk_create_cuda_module(CudaImageFilters
    DEPENDS MitkCore
    CPP_FILES ${CPP_FILES} ${CU_FILES}
    H_FILES ${H_FILES} ${CUH_FILES}
)

if(BUILD_TESTING)
    # MUT double with fake/mocked cuda runtime for testing
    mitk_create_cuda_module_mut(CudaImageFilters
        DEPENDS MitkCore MitkMockCudaRuntime
        CPP_FILES ${CPP_FILES}
        H_FILES ${HEADER_FILES}
    )

    # unit tests with hardware
    add_subdirectory(test)
    # unit tests with mocked hardware
    add_subdirectory(mock_test)
endif()
