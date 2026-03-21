cmake_minimum_required(VERSION 3.20)
project(druse_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "arm64")

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

# Build static library
add_library(druse_core STATIC druse_core.cpp druse_2d.cpp)

target_include_directories(druse_core PRIVATE
    ${lbfgspp_SOURCE_DIR}/include
)

target_link_libraries(druse_core
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
