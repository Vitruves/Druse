cmake_minimum_required(VERSION 3.20)
project(druse_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "arm64")

# Xcode links the app with Apple's libLTO. If CMake builds static archives
# with a newer non-Apple LLVM and ThinLTO enabled, the final app link can fail
# with "could not parse bitcode object file". Keep the C++ core and vendored
# gemmi as plain object code.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF CACHE BOOL "" FORCE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO OFF CACHE BOOL "" FORCE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL OFF CACHE BOOL "" FORCE)

# RDKit (Homebrew)
set(RDKIT_ROOT "/opt/homebrew" CACHE PATH "RDKit root directory")
set(BOOST_ROOT "/opt/homebrew/opt/boost" CACHE PATH "Boost root directory")

include_directories(${RDKIT_ROOT}/include/rdkit)
include_directories(${BOOST_ROOT}/include)
link_directories(${RDKIT_ROOT}/lib)

# Eigen3 (header-only, Homebrew) — Eigen 5 is backwards-compatible
include_directories(/opt/homebrew/include/eigen3)

# nanoflann (header-only, Homebrew)
include_directories(/opt/homebrew/include)

# TBB
find_package(TBB REQUIRED)

# LBFGSpp — header-only, just download headers (skip its CMake which requires Eigen3 cmake target)
include(FetchContent)
FetchContent_Declare(LBFGSpp
    GIT_REPOSITORY https://github.com/yixuan/LBFGSpp.git
    GIT_TAG v0.3.0
)
FetchContent_Populate(LBFGSpp)

# gemmi — fetched at configure time, built as static lib
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_GEMMI_PROGRAM OFF CACHE BOOL "" FORCE)
set(INSTALL_DEV_FILES OFF CACHE BOOL "" FORCE)
set(USE_PYTHON OFF CACHE BOOL "" FORCE)
set(INTERNAL_ZLIB ON CACHE BOOL "" FORCE)

FetchContent_Declare(gemmi
    GIT_REPOSITORY https://github.com/project-gemmi/gemmi.git
    GIT_TAG        v0.7.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(gemmi)

set_target_properties(gemmi_cpp PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    INTERPROCEDURAL_OPTIMIZATION FALSE
)
target_compile_options(gemmi_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-fno-lto>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-lto>
)

# OpenMM (optional — auto-fetched and built at configure time)
option(DRUSE_ENABLE_OPENMM "Build with OpenMM refinement support" ON)

set(OPENMM_SRC_DIR    "${FETCHCONTENT_BASE_DIR}/openmm-src")
set(OPENMM_BUILD_DIR  "${FETCHCONTENT_BASE_DIR}/openmm-build")
set(OPENMM_INSTALL_DIR "${FETCHCONTENT_BASE_DIR}/openmm-install")

if(DRUSE_ENABLE_OPENMM)
    if(NOT EXISTS "${OPENMM_INSTALL_DIR}/lib/libOpenMM.dylib")
        message(STATUS "OpenMM not found — fetching and building (first time only, may take a few minutes)...")

        # Clone source via FetchContent
        FetchContent_Declare(openmm
            GIT_REPOSITORY https://github.com/openmm/openmm.git
            GIT_TAG        8.5.0
            GIT_SHALLOW    TRUE
        )
        FetchContent_Populate(openmm)

        # Configure, build, and install as a standalone project
        execute_process(
            COMMAND ${CMAKE_COMMAND}
                -G "${CMAKE_GENERATOR}"
                -S "${OPENMM_SRC_DIR}"
                -B "${OPENMM_BUILD_DIR}"
                -DCMAKE_INSTALL_PREFIX=${OPENMM_INSTALL_DIR}
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_OSX_ARCHITECTURES=arm64
                -DBUILD_TESTING=OFF
                -DOPENMM_BUILD_PYTHON_WRAPPERS=OFF
                -DOPENMM_BUILD_C_AND_FORTRAN_WRAPPERS=OFF
                -DOPENMM_BUILD_EXAMPLES=OFF
                -DCMAKE_C_FLAGS=-w
                -DCMAKE_CXX_FLAGS=-w
            RESULT_VARIABLE _openmm_cfg_result
        )
        if(_openmm_cfg_result EQUAL 0)
            execute_process(
                COMMAND ${CMAKE_COMMAND} --build "${OPENMM_BUILD_DIR}" --config Release -j8
                RESULT_VARIABLE _openmm_build_result
            )
        endif()
        if(_openmm_cfg_result EQUAL 0 AND _openmm_build_result EQUAL 0)
            execute_process(
                COMMAND ${CMAKE_COMMAND} --install "${OPENMM_BUILD_DIR}" --config Release
            )
        else()
            message(WARNING "OpenMM build failed — building without OpenMM refinement")
        endif()
    endif()
endif()

if(DRUSE_ENABLE_OPENMM AND EXISTS "${OPENMM_INSTALL_DIR}/lib/libOpenMM.dylib")
    set(DRUSE_HAS_OPENMM ON)
    message(STATUS "OpenMM found at ${OPENMM_INSTALL_DIR}")
else()
    set(DRUSE_HAS_OPENMM OFF)
    message(STATUS "OpenMM not available — building without OpenMM refinement")
endif()

# Build static library
add_library(druse_core STATIC
    druse_core.cpp
    druse_core_ionization.cpp
    druse_core_variants.cpp
    druse_core_io.cpp
    druse_core_geometry.cpp
    druse_core_analysis.cpp
    druse_core_protonation_tools.cpp
    druse_core_pka_featurize.cpp
    druse_2d.cpp
    druse_gemmi.cpp
    druse_openmm.cpp
    druse_xtb.cpp
)
set_target_properties(druse_core PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION FALSE
)
target_compile_options(druse_core PRIVATE -fno-lto)
target_compile_definitions(druse_core PRIVATE ACCELERATE_NEW_LAPACK)

target_include_directories(druse_core PRIVATE
    ${lbfgspp_SOURCE_DIR}/include
)

if(DRUSE_HAS_OPENMM)
    target_compile_definitions(druse_core PRIVATE DRUSE_HAS_OPENMM)
    target_include_directories(druse_core SYSTEM PRIVATE "${OPENMM_INSTALL_DIR}/include")
    target_link_directories(druse_core PRIVATE "${OPENMM_INSTALL_DIR}/lib")
    target_link_libraries(druse_core OpenMM)
endif()

target_link_libraries(druse_core
    "-framework Accelerate"
    gemmi_cpp
    TBB::tbb
    RDKitChemicalFeatures
    RDKitSmilesParse
    RDKitGraphMol
    RDKitRDGeneral
    RDKitDistGeomHelpers
    RDKitDistGeometry
    RDKitForceFieldHelpers
    RDKitForceField
    RDKitMMPA
    RDKitPartialCharges
    RDKitDescriptors
    RDKitMolTransforms
    RDKitSubstructMatch
    RDKitEigenSolvers
    RDKitAlignment
    RDKitRDGeometryLib
    RDKitDataStructs
    RDKitDepictor
    RDKitcoordgen
    RDKitMolDraw2D
    RDKitFileParsers
    RDKitChemTransforms
    RDKitMolStandardize
    RDKitFingerprints
    RDKitGenericGroups
    RDKitMolChemicalFeatures
    RDKitRingDecomposerLib
    RDKitFMCS
)

install(TARGETS druse_core ARCHIVE DESTINATION lib)
install(FILES druse_core.h DESTINATION include)
