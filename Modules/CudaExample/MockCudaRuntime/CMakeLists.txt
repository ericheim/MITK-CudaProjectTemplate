set(src_files
    cuda_runtime.h
    mock_cuda_runtime.h
    mock_device_types.h

    cuda_runtime.cpp
)

set(name MitkMockCudaRuntime)

add_library(${name} ${src_files})

target_link_libraries(${name}
    PUBLIC
        GTest::gmock
)

target_include_directories(${name}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
