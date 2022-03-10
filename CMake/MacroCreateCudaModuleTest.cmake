# ============================================================================
# Copyright (c) Eric Heim
# All rights reserved.
#
# Use of this source code is governed by a 3-clause BSD license that can be
# found in the LICENSE file.
#
# ============================================================================

##############################################################################
#   Macro to create cuda enable google tests for MITK modules.
#   PARAMS:
#       WITH_CUDA - Link cuda runtime and include cuda headers in the test
#                   driver.This enables calling cuda api functions within
#                   the test.
#       DEPENDS   - Additional dependencies to be used in the test.
#       CPP_FILES - C++ sources
#       H_FILES   - Headers
##############################################################################
macro(create_cuda_module_test module_name)
    cmake_parse_arguments(CUDA_MODULE_TEST "WITH_CUDA" "" "DEPENDS;CPP_FILES;H_FILES" ${ARGN})

    set(test_name Test${module_name})

    add_executable(${test_name} ${CUDA_MODULE_TEST_H_FILES} ${CUDA_MODULE_TEST_CPP_FILES})

    if(CUDA_MODULE_TEST_WITH_CUDA)
        set(cudart_lib CUDA::cudart)

        set_target_properties(${test_name}
            PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                POSITION_INDEPENDENT_CODE ON
        )

        target_include_directories(${test_name}
            PRIVATE
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
    endif()

    target_link_libraries(${test_name}
        PRIVATE
            GTest::gtest
            GTest::gmock
            GTest::gtest_main
            ${cudart_lib}
            ${module_name}
            ${CUDA_MODULE_TEST_DEPENDS}
    )

    if(APPLE)
        set_target_properties(${test_name} PROPERTIES INSTALL_RPATH "@loader_path/../lib")
    elseif(UNIX AND NOT APPLE)
        set_target_properties(${test_name} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
    endif()

    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()
