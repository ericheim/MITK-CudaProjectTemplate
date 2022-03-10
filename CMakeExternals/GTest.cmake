
if(BUILD_TESTING)
    set(proj GTest)
    set(proj_DEPENDENCIES "")

    if(MITK_USE_${proj})
        set(${proj}_DEPENDS ${proj})

        if(DEFINED ${proj}_DIR AND NOT EXISTS ${${proj}_DIR})
            message(FATAL_ERROR "${proj}_DIR variable is defined but corresponds to non-existing directory!")
        endif()

        ExternalProject_Add(
            ${proj}
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.11.0
            CMAKE_ARGS ${ep_common_args}
            CMAKE_CACHE_ARGS ${ep_common_cache_args}
            CMAKE_CACHE_DEFAULT_ARGS ${ep_common_cache_default_args}
            DEPENDS ${proj_DEPENDENCIES}
        )
        set(${proj}_DIR ${ep_prefix})
    else()
        mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
    endif()
endif()
