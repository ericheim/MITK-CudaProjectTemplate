
# ============================================================================
# Copyright (c) Eric Heim
# All rights reserved.
#
# Use of this source code is governed by a 3-clause BSD license that can be
# found in the LICENSE file.
#
# ============================================================================


##############################################################################
#   Wrapper function arround MitkCreateModule to enable modules with
#   cuda support. The function only supports a subset of the MITK_CREATE_MODULE
#   parameters that are passed on to the original function.
#
#   enable_language(CUDA) - has to be activated to use the function
#   find_package(CUDAToolkit) - is required for linking of cudart
#
#   PARAMS:
#       DEPENDS   - Additional dependencies to be used in the test.
#       CPP_FILES - C++ sources
#       H_FILES   - Headers
#       CU_FILES  - CUDA files compiled with nvcc
##############################################################################
function(mitk_create_cuda_module)
    list(GET ARGN 0 module_name)
    set(name Mitk${module_name})

    cmake_parse_arguments(CUDA_MODULE "" "" "DEPENDS;CPP_FILES;H_FILES;CU_FILES" ${ARGN})

    mitk_create_module(${module_name}
        CPP_FILES ${CUDA_MODULE_CPP_FILES} ${CUDA_MODULE_CU_FILES}
        H_FILES ${CUDA_MODULE_H_FILES}
        DEPENDS PUBLIC ${CUDA_MODULE_DEPENDS}
    )

    target_include_directories(${name}
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    target_link_libraries(${name}
        PRIVATE CUDA::cudart
    )

    set_target_properties(${name}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-std=c++11>
    )
endfunction()

##############################################################################
#   Stripped down module without CUDA to create a MUT
function(mitk_create_cuda_module_mut)
    list(GET ARGN 0 module_name)

    cmake_parse_arguments(MUT_MODULE "" "" "DEPENDS;CPP_FILES;H_FILES" ${ARGN})

    mitk_create_module(${module_name}_MUT
        CPP_FILES ${MUT_MODULE_CPP_FILES}
        H_FILES ${MUT_MODULE_H_FILES}
        DEPENDS PUBLIC ${MUT_MODULE_DEPENDS}
    )

endfunction()
