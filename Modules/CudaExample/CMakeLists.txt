include(FunctionCreateCudaModule)
include(MacroCreateCudaModuleTest)

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    add_subdirectory(MockCudaRuntime)
endif()

add_subdirectory(CudaImageFilters)
