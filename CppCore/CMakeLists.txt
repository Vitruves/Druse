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

# gemmi
set(GEMMI_VENDORED_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../third_parties_deps/gemmi")
set(GEMMI_LEGACY_TEMP_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../temp/gemmi")
set(GEMMI_SOURCE_DIR "" CACHE PATH "Path to a local gemmi checkout override")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_GEMMI_PROGRAM OFF CACHE BOOL "" FORCE)
set(INSTALL_DEV_FILES OFF CACHE BOOL "" FORCE)
set(USE_PYTHON OFF CACHE BOOL "" FORCE)
set(INTERNAL_ZLIB ON CACHE BOOL "" FORCE)

if (GEMMI_SOURCE_DIR STREQUAL "${GEMMI_LEGACY_TEMP_SOURCE_DIR}")
    set(GEMMI_SOURCE_DIR "" CACHE PATH "Path to a local gemmi checkout override" FORCE)
endif()

set(GEMMI_ACTIVE_SOURCE_DIR "${GEMMI_VENDORED_SOURCE_DIR}")
if (GEMMI_SOURCE_DIR)
    set(GEMMI_ACTIVE_SOURCE_DIR "${GEMMI_SOURCE_DIR}")
endif()

if (NOT EXISTS "${GEMMI_ACTIVE_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Gemmi source not found at ${GEMMI_ACTIVE_SOURCE_DIR}. "
        "Vendor it under third_parties_deps/gemmi or pass -DGEMMI_SOURCE_DIR=/path/to/gemmi."
    )
endif()

add_subdirectory("${GEMMI_ACTIVE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/gemmi-build-vendored" EXCLUDE_FROM_ALL)

set_target_properties(gemmi_cpp PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    INTERPROCEDURAL_OPTIMIZATION FALSE
)
target_compile_options(gemmi_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-fno-lto>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-lto>
)

# Build static library
add_library(druse_core STATIC druse_core.cpp druse_2d.cpp druse_gemmi.cpp)
set_target_properties(druse_core PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION FALSE
)
target_compile_options(druse_core PRIVATE -fno-lto)

target_include_directories(druse_core PRIVATE
    ${lbfgspp_SOURCE_DIR}/include
)

target_link_libraries(druse_core
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
    RDKitFingerprints
    RDKitGenericGroups
    RDKitMolChemicalFeatures
    RDKitRingDecomposerLib
)

install(TARGETS druse_core ARCHIVE DESTINATION lib)
install(FILES druse_core.h DESTINATION include)
