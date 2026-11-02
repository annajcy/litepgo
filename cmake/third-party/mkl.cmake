# Intel oneMKL provides the PARDISO sparse backends on x86-64 Linux and Windows through
# <Eigen/PardisoSupport>. MKL is linked dynamically with its oneTBB threading
# layer, which runs on litepgo's own TBB runtime: MKL 2026.1 is built against
# tbb==2023.*, matching the fetched oneTBB 2023.1. tbb::global_control and
# task_arena therefore bound PARDISO like every other litepgo parallel loop.
# Eigen dense code does not route through MKL.
#
# By default pinned wheels from PyPI are downloaded and
# unpacked once into LITEPGO_MKL_CACHE_DIR, which every build tree shares.
# LITEPGO_MKL_ROOT selects a local oneMKL 2026.1 install instead, such as
# /opt/intel/oneapi/mkl/2026.1.
add_library(litepgo_mkl INTERFACE)
add_library(litepgo::mkl ALIAS litepgo_mkl)

if(LITEPGO_WITH_MKL)
    set(LITEPGO_MKL_ROOT "" CACHE PATH "Local oneMKL 2026.1 install to use instead of the fetched wheels")
    set(LITEPGO_MKL_CACHE_DIR "${PROJECT_SOURCE_DIR}/build/_cache" CACHE PATH
        "Where the oneMKL wheels are downloaded and unpacked, shared by every build tree")

    # Downloads a wheel (a zip archive) and unpacks its members matching the
    # remaining arguments into destination. A stamp named after the hash marks a
    # finished unpack; the lock keeps concurrent configures from racing.
    function(litepgo_unpack_wheel url sha256 destination)
        set(stamp "${destination}/.unpacked-${sha256}")
        if(EXISTS "${stamp}")
            return()
        endif()
        file(MAKE_DIRECTORY "${LITEPGO_MKL_CACHE_DIR}")
        file(LOCK "${LITEPGO_MKL_CACHE_DIR}/.lock" GUARD FUNCTION)
        if(EXISTS "${stamp}")
            return()
        endif()
        get_filename_component(archive "${url}" NAME)
        set(archive "${LITEPGO_MKL_CACHE_DIR}/${archive}")
        if(EXISTS "${archive}")
            file(SHA256 "${archive}" actual)
        endif()
        if(NOT actual STREQUAL sha256)
            message(STATUS "Downloading ${url}")
            file(DOWNLOAD "${url}" "${archive}" EXPECTED_HASH SHA256=${sha256} STATUS status)
            list(GET status 0 code)
            if(NOT code EQUAL 0)
                file(REMOVE "${archive}")
                message(FATAL_ERROR "Downloading ${url} failed: ${status}")
            endif()
        endif()
        file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${destination}" PATTERNS ${ARGN})
        file(TOUCH "${stamp}")
    endfunction()

    if(LITEPGO_MKL_ROOT)
        set(mkl_include_dir "${LITEPGO_MKL_ROOT}/include")
        set(mkl_lib_dir "${LITEPGO_MKL_ROOT}/lib")
        set(mkl_bin_dir "${LITEPGO_MKL_ROOT}/bin")
    elseif(WIN32)
        set(mkl_prefix "${LITEPGO_MKL_CACHE_DIR}/mkl-2026.1.0-windows")
        litepgo_unpack_wheel(
            https://files.pythonhosted.org/packages/48/2e/8c71366e2f35ba74704b68be65caec6674472e7913ead5196cc69a15001e/mkl_include-2026.1.0-py2.py3-none-win_amd64.whl
            d933db23875da7d145a8897fa0cab1014228af3c387972f25b65472b49b23c94
            "${mkl_prefix}" "*/include/*")
        litepgo_unpack_wheel(
            https://files.pythonhosted.org/packages/ef/d4/6702825beac5860fa7785eaca941541968fd5ce8c49fed88c10bb15b1f69/mkl_devel-2026.1.0-py2.py3-none-win_amd64.whl
            ead22e4683f121732ef9602de695496106ed01f452f1bea72bc8df71cc79a43c
            "${mkl_prefix}" "*/lib/mkl_intel_lp64_dll.lib" "*/lib/mkl_tbb_thread_dll.lib" "*/lib/mkl_core_dll.lib")
        litepgo_unpack_wheel(
            https://files.pythonhosted.org/packages/d6/bd/e35bd5477278c91b0f5d4733b13009d183c284d636d68c649204858c207a/mkl-2026.1.0-py2.py3-none-win_amd64.whl
            8194a8270d70622eff00eec84c793efc382b14848b98c3ace6bacdda42170a80
            "${mkl_prefix}"
            "*/bin/libimalloc.dll" "*/bin/mkl_tbb_thread.3.dll" "*/bin/mkl_core.3.dll"
            "*/bin/mkl_def.3.dll" "*/bin/mkl_mc3.3.dll" "*/bin/mkl_avx*.3.dll" "*/bin/mkl_vml_*.3.dll")
        set(mkl_include_dir "${mkl_prefix}/mkl_include-2026.1.0.data/data/Library/include")
        set(mkl_lib_dir "${mkl_prefix}/mkl_devel-2026.1.0.data/data/Library/lib")
        set(mkl_bin_dir "${mkl_prefix}/mkl-2026.1.0.data/data/Library/bin")
    else()
        set(mkl_prefix "${LITEPGO_MKL_CACHE_DIR}/mkl-2026.1.0")
        litepgo_unpack_wheel(
            https://files.pythonhosted.org/packages/5c/42/76a2c4964399376855ed3d4867ad2c2c01c9a7297dca307ce43c1ef76c71/mkl_include-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl
            e64b260e67284b8a696343d58c339ad9d990cc985077dab1269d1c6d2a2e2b1b
            "${mkl_prefix}" "*/include/*")
        # libmkl_core loads the CPU-specific kernels (def, mc3, avx2, ...) from its
        # own directory at run time.
        litepgo_unpack_wheel(
            https://files.pythonhosted.org/packages/61/da/4921e17b1f455f7fed30d5cc0964f3289eee6a6cb03cdf7d5e20c14bd025/mkl-2026.1.0-py2.py3-none-manylinux_2_28_x86_64.whl
            4d5a26449818a8aebd4b2aaf6b291831e691f9cf74b152df49b99cf48bbd5360
            "${mkl_prefix}"
            "*/lib/libmkl_intel_lp64.so.3" "*/lib/libmkl_tbb_thread.so.3" "*/lib/libmkl_core.so.3"
            "*/lib/libmkl_def.so.3" "*/lib/libmkl_mc3.so.3" "*/lib/libmkl_avx*.so.3" "*/lib/libmkl_vml_*.so.3")
        set(mkl_include_dir "${mkl_prefix}/mkl_include-2026.1.0.data/data/include")
        set(mkl_lib_dir "${mkl_prefix}/mkl-2026.1.0.data/data/lib")
    endif()

    if(NOT EXISTS "${mkl_include_dir}/mkl_version.h")
        message(FATAL_ERROR "oneMKL headers not found in ${mkl_include_dir}; "
                            "fix LITEPGO_MKL_ROOT or configure with -DLITEPGO_WITH_MKL=OFF")
    endif()
    file(STRINGS "${mkl_include_dir}/mkl_version.h" mkl_version REGEX "^#define INTEL_MKL_VERSION ")
    if(NOT mkl_version MATCHES " 20260100$")
        message(FATAL_ERROR "litepgo needs oneMKL 2026.1, which matches its oneTBB 2023.1; "
                            "${mkl_include_dir} has ${mkl_version}")
    endif()

    # Linked in the order the MKL link line advisor gives: interface, threading, core.
    foreach(name mkl_intel_lp64 mkl_tbb_thread mkl_core)
        if(WIN32)
            set(library "${mkl_lib_dir}/${name}_dll.lib")
            # The LP64 interface is an archive, while the other .libs import DLLs.
            add_library(litepgo::${name} UNKNOWN IMPORTED)
            set_target_properties(litepgo::${name} PROPERTIES IMPORTED_LOCATION "${library}")
        else()
            set(library "${mkl_lib_dir}/lib${name}.so.3")
            add_library(litepgo::${name} SHARED IMPORTED)
            set_target_properties(litepgo::${name} PROPERTIES IMPORTED_LOCATION "${library}" IMPORTED_SONAME "lib${name}.so.3")
        endif()
        if(NOT EXISTS "${library}")
            message(FATAL_ERROR "${library} not found; fix LITEPGO_MKL_ROOT or configure with -DLITEPGO_WITH_MKL=OFF")
        endif()
        target_link_libraries(litepgo_mkl INTERFACE litepgo::${name})
    endforeach()
    target_link_libraries(litepgo_mkl INTERFACE TBB::tbb)
    if(WIN32)
        foreach(name libimalloc mkl_tbb_thread.3 mkl_core.3 mkl_def.3 mkl_mc3.3)
            if(NOT EXISTS "${mkl_bin_dir}/${name}.dll")
                message(FATAL_ERROR "${mkl_bin_dir}/${name}.dll not found; fix LITEPGO_MKL_ROOT")
            endif()
        endforeach()
        file(GLOB LITEPGO_MKL_RUNTIME_DLLS "${mkl_bin_dir}/libimalloc.dll"
            "${mkl_bin_dir}/mkl_tbb_thread.3.dll" "${mkl_bin_dir}/mkl_core.3.dll"
            "${mkl_bin_dir}/mkl_def.3.dll" "${mkl_bin_dir}/mkl_mc3.3.dll"
            "${mkl_bin_dir}/mkl_avx*.3.dll" "${mkl_bin_dir}/mkl_vml_*.3.dll")
    else()
        # Keep TBB a direct dependency so the executable's RUNPATH also resolves MKL's runtime.
        target_link_options(litepgo_mkl INTERFACE "LINKER:--no-as-needed")
    endif()
    target_include_directories(litepgo_mkl SYSTEM INTERFACE "${mkl_include_dir}")
    target_compile_definitions(litepgo_mkl INTERFACE LITEPGO_HAS_MKL)
    message(STATUS "oneMKL 2026.1 PARDISO from ${mkl_lib_dir}")
endif()
