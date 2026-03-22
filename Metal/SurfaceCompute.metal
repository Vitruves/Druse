#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// ============================================================================
// MARK: - Surface Compute Types
// ============================================================================

struct SurfaceVertex {
    packed_float3 position;
    packed_float3 normal;
    float4        color;
};

struct SurfaceAtom {
    packed_float3 position;
    float         vdwRadius;
    float         charge;
    uint          atomicNum;
    uint          isAromatic;  // 1 if aromatic ring member
    float         _pad0;
};

struct SurfaceGridParams {
    packed_float3 origin;
    float         spacing;
    uint3         dims;       // nx, ny, nz
    uint          totalPoints;
    float         isovalue;
    uint          numAtoms;
    uint          maxVertices;
    uint          maxIndices;
    float         probeRadius; // 1.4 Angstroms for water
    float         _pad0;
    float         _pad1;
    float         _pad2;
};

// ============================================================================
// MARK: - Marching Cubes Lookup Tables
// ============================================================================

// Edge table: for each of 256 cube configurations, bitmask of which edges are intersected
constant uint edgeTable[256] = {
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

// Triangle table: for each of 256 cube configurations, list of edge triplets forming triangles.
// -1 terminates the list. Maximum 5 triangles (15 edges) per configuration.
constant int triTable[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1, -1},
    { 3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1, -1},
    { 3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    { 3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1},
    { 9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1},
    { 2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1, -1},
    { 8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1},
    { 4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1, -1},
    { 3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1, -1},
    { 1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1, -1},
    { 4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1, -1},
    { 4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1},
    { 5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1, -1},
    { 2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1, -1},
    { 9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1},
    { 0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1},
    { 2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1, -1},
    {10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1, -1},
    { 5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1, -1},
    { 5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1},
    { 9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1},
    { 1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1, -1},
    {10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1, -1},
    { 8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1, -1},
    { 2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1},
    { 7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1, -1},
    { 2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1, -1},
    {11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1, -1},
    { 5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0, -1},
    {11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0, -1},
    {11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1, -1},
    { 9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1, -1},
    { 2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1, -1},
    { 6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1, -1},
    { 3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1, -1},
    { 6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1},
    {10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1, -1},
    { 6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1, -1},
    { 8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1, -1},
    { 7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9, -1},
    { 3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1},
    { 5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1, -1},
    { 0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1},
    { 9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6, -1},
    { 8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1, -1},
    { 5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11, -1},
    { 0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7, -1},
    { 6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1, -1},
    {10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1},
    {10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1},
    { 8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1, -1},
    { 1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1, -1},
    { 0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1},
    {10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1, -1},
    { 3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1, -1},
    { 6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1, -1},
    { 9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1, -1},
    { 8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1, -1},
    { 3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1, -1},
    { 6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1, -1},
    { 0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1, -1},
    {10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1, -1},
    {10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1, -1},
    { 2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9, -1},
    { 7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1, -1},
    { 7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1, -1},
    { 2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7, -1},
    { 1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11, -1},
    {11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1, -1},
    { 8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6, -1},
    { 0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1, -1},
    { 7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1},
    {10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1},
    { 2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1},
    { 6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1, -1},
    { 7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1, -1},
    { 2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1, -1},
    { 1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1, -1},
    {10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1, -1},
    {10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1, -1},
    { 0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1, -1},
    { 7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1, -1},
    { 6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1, -1},
    { 8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1, -1},
    { 9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1, -1},
    { 6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1, -1},
    { 4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1, -1},
    {10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3, -1},
    { 8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1},
    { 0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1, -1},
    { 1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1, -1},
    { 8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1, -1},
    {10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1, -1},
    { 4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3, -1},
    {10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1},
    { 5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1},
    {11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1, -1},
    { 9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1},
    { 6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1, -1},
    { 7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1, -1},
    { 3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6, -1},
    { 7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1, -1},
    { 3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1, -1},
    { 6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8, -1},
    { 9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1, -1},
    { 1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4, -1},
    { 4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10, -1},
    { 7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1, -1},
    { 6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1, -1},
    { 3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1, -1},
    { 0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1, -1},
    { 6, 11,  3,  6,  3,  5,  5,  3,  1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1, -1},
    { 0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10, -1},
    {11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5, -1},
    { 6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1, -1},
    { 5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1, -1},
    { 9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1, -1},
    { 1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8, -1},
    { 1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6, -1},
    {10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, -1, -1, -1, -1},
    { 0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1, -1},
    { 5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1, -1},
    {10,  7,  5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1, -1},
    {11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1, -1},
    { 9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1, -1},
    { 7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2, -1},
    { 2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1},
    { 8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1, -1},
    { 9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, -1, -1, -1, -1},
    { 9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2, -1},
    { 1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1, -1},
    { 9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1, -1},
    { 9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1},
    { 5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1, -1},
    { 0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1, -1},
    {10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4, -1},
    { 2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1, -1},
    { 0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5,  1, 11, -1},
    { 0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5, -1},
    { 9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1, -1},
    { 5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1, -1},
    { 3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9, -1},
    { 5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, -1, -1, -1, -1},
    { 8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, -1, -1, -1, -1},
    { 9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    { 0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1, -1},
    { 1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1, -1},
    { 3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4, -1},
    { 4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1, -1},
    { 9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3, -1},
    {11,  7,  4,  11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1, -1},
    {11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1, -1},
    { 2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1, -1},
    { 9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7, -1},
    { 3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10, -1},
    { 1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  1,  4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1, -1},
    { 4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1, -1},
    { 4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 9, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1, -1},
    { 0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
    { 3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1, -1},
    { 3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1, -1},
    { 0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1, -1},
    { 9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1, -1},
    { 1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

// ============================================================================
// MARK: - Signed Distance Field for Connolly / SES Surface
// ============================================================================

/// Compute signed distance field (SDF) for Solvent Excluded Surface (Connolly surface).
/// SDF value at point P = min over all atoms of (|P - atom_center| - vdw_radius - probe_radius).
/// Negative = inside the molecular surface, positive = outside. Isosurface at 0.
/// Each thread processes one grid point.
kernel void computeDistanceField(
    device float               *distanceField [[buffer(0)]],
    constant SurfaceAtom       *atoms         [[buffer(1)]],
    constant SurfaceGridParams &params        [[buffer(2)]],
    uint                        tid           [[thread_position_in_grid]])
{
    if (tid >= params.totalPoints) return;

    uint nx = params.dims.x;
    uint ny = params.dims.y;
    uint iz = tid / (nx * ny);
    uint iy = (tid - iz * nx * ny) / nx;
    uint ix = tid - iz * nx * ny - iy * nx;

    float3 gridPos = float3(params.origin) + float3(float(ix), float(iy), float(iz)) * params.spacing;

    float probe = params.probeRadius;
    float minDist = 1e10f;

    for (uint i = 0; i < params.numAtoms; i++) {
        float3 atomPos = float3(atoms[i].position);
        float dist = length(gridPos - atomPos);
        float sdf = dist - atoms[i].vdwRadius - probe;
        minDist = min(minDist, sdf);
    }

    distanceField[tid] = minDist;
}

// ============================================================================
// MARK: - Gaussian Field Computation
// ============================================================================

/// Compute Gaussian scalar field for molecular surface.
/// Each thread processes one grid point.
kernel void computeGaussianField(
    device float               *scalarField  [[buffer(0)]],
    constant SurfaceAtom       *atoms        [[buffer(1)]],
    constant SurfaceGridParams &params       [[buffer(2)]],
    uint                        tid          [[thread_position_in_grid]])
{
    if (tid >= params.totalPoints) return;

    uint nx = params.dims.x;
    uint ny = params.dims.y;
    uint iz = tid / (nx * ny);
    uint iy = (tid - iz * nx * ny) / nx;
    uint ix = tid - iz * nx * ny - iy * nx;

    float3 gridPos = float3(params.origin) + float3(float(ix), float(iy), float(iz)) * params.spacing;

    float fieldValue = 0.0f;
    float alpha = 2.0f;
    float cutoff = 8.0f;
    float cutoffSq = cutoff * cutoff;

    for (uint i = 0; i < params.numAtoms; i++) {
        float3 atomPos = float3(atoms[i].position);
        float3 diff = gridPos - atomPos;
        float distSq = dot(diff, diff);

        if (distSq > cutoffSq) continue;

        float dist = sqrt(distSq);
        float vdwR = atoms[i].vdwRadius;
        float dr = dist - vdwR;
        fieldValue += exp(-alpha * dr * dr);
    }

    scalarField[tid] = fieldValue;
}

// ============================================================================
// MARK: - Marching Cubes
// ============================================================================

/// Helper: interpolate vertex position along edge between two grid points.
inline float3 interpolateEdge(float3 p1, float3 p2, float v1, float v2, float iso) {
    float t = (iso - v1) / (v2 - v1 + 1e-10f);
    t = clamp(t, 0.0f, 1.0f);
    return mix(p1, p2, t);
}

/// Compute gradient (normal) at a point in the scalar field using central differences.
inline float3 computeGradient(
    device const float *field,
    uint ix, uint iy, uint iz,
    uint nx, uint ny, uint nz)
{
    // Clamp to avoid out-of-bounds
    uint ixm = ix > 0 ? ix - 1 : 0;
    uint ixp = ix < nx - 1 ? ix + 1 : nx - 1;
    uint iym = iy > 0 ? iy - 1 : 0;
    uint iyp = iy < ny - 1 ? iy + 1 : ny - 1;
    uint izm = iz > 0 ? iz - 1 : 0;
    uint izp = iz < nz - 1 ? iz + 1 : nz - 1;

    float dx = field[iz * nx * ny + iy * nx + ixp] - field[iz * nx * ny + iy * nx + ixm];
    float dy = field[iz * nx * ny + iyp * nx + ix] - field[iz * nx * ny + iym * nx + ix];
    float dz = field[izp * nx * ny + iy * nx + ix] - field[izm * nx * ny + iy * nx + ix];

    float3 grad = float3(dx, dy, dz);
    float len = length(grad);
    return len > 1e-8f ? -grad / len : float3(0.0f, 1.0f, 0.0f);
}

/// Marching cubes kernel: each thread processes one voxel.
kernel void marchingCubes(
    device const float         *scalarField    [[buffer(0)]],
    device SurfaceVertex       *vertices       [[buffer(1)]],
    device uint                *indices        [[buffer(2)]],
    device atomic_uint         *vertexCounter  [[buffer(3)]],
    device atomic_uint         *indexCounter   [[buffer(4)]],
    constant SurfaceGridParams &params         [[buffer(5)]],
    uint                        tid            [[thread_position_in_grid]])
{
    uint nx = params.dims.x;
    uint ny = params.dims.y;
    uint nz = params.dims.z;

    // We process (nx-1) * (ny-1) * (nz-1) voxels
    uint voxelNx = nx - 1;
    uint voxelNy = ny - 1;
    uint voxelNz = nz - 1;
    uint totalVoxels = voxelNx * voxelNy * voxelNz;

    if (tid >= totalVoxels) return;

    uint iz = tid / (voxelNx * voxelNy);
    uint iy = (tid - iz * voxelNx * voxelNy) / voxelNx;
    uint ix = tid - iz * voxelNx * voxelNy - iy * voxelNx;

    float iso = params.isovalue;

    // 8 corner values of this voxel
    float v[8];
    v[0] = scalarField[ iz      * nx * ny +  iy      * nx + ix    ];
    v[1] = scalarField[ iz      * nx * ny +  iy      * nx + ix + 1];
    v[2] = scalarField[ iz      * nx * ny + (iy + 1) * nx + ix + 1];
    v[3] = scalarField[ iz      * nx * ny + (iy + 1) * nx + ix    ];
    v[4] = scalarField[(iz + 1) * nx * ny +  iy      * nx + ix    ];
    v[5] = scalarField[(iz + 1) * nx * ny +  iy      * nx + ix + 1];
    v[6] = scalarField[(iz + 1) * nx * ny + (iy + 1) * nx + ix + 1];
    v[7] = scalarField[(iz + 1) * nx * ny + (iy + 1) * nx + ix    ];

    // Build cube index from corner signs
    uint cubeIndex = 0;
    if (v[0] >= iso) cubeIndex |= 1;
    if (v[1] >= iso) cubeIndex |= 2;
    if (v[2] >= iso) cubeIndex |= 4;
    if (v[3] >= iso) cubeIndex |= 8;
    if (v[4] >= iso) cubeIndex |= 16;
    if (v[5] >= iso) cubeIndex |= 32;
    if (v[6] >= iso) cubeIndex |= 64;
    if (v[7] >= iso) cubeIndex |= 128;

    if (edgeTable[cubeIndex] == 0) return;

    // Corner positions in world space
    float3 origin = float3(params.origin);
    float sp = params.spacing;
    float3 p[8];
    p[0] = origin + float3(float(ix),     float(iy),     float(iz))     * sp;
    p[1] = origin + float3(float(ix + 1), float(iy),     float(iz))     * sp;
    p[2] = origin + float3(float(ix + 1), float(iy + 1), float(iz))     * sp;
    p[3] = origin + float3(float(ix),     float(iy + 1), float(iz))     * sp;
    p[4] = origin + float3(float(ix),     float(iy),     float(iz + 1)) * sp;
    p[5] = origin + float3(float(ix + 1), float(iy),     float(iz + 1)) * sp;
    p[6] = origin + float3(float(ix + 1), float(iy + 1), float(iz + 1)) * sp;
    p[7] = origin + float3(float(ix),     float(iy + 1), float(iz + 1)) * sp;

    // Compute edge intersection positions (up to 12 edges)
    float3 edgeVerts[12];
    uint edges = edgeTable[cubeIndex];
    if (edges &    1) edgeVerts[0]  = interpolateEdge(p[0], p[1], v[0], v[1], iso);
    if (edges &    2) edgeVerts[1]  = interpolateEdge(p[1], p[2], v[1], v[2], iso);
    if (edges &    4) edgeVerts[2]  = interpolateEdge(p[2], p[3], v[2], v[3], iso);
    if (edges &    8) edgeVerts[3]  = interpolateEdge(p[3], p[0], v[3], v[0], iso);
    if (edges &   16) edgeVerts[4]  = interpolateEdge(p[4], p[5], v[4], v[5], iso);
    if (edges &   32) edgeVerts[5]  = interpolateEdge(p[5], p[6], v[5], v[6], iso);
    if (edges &   64) edgeVerts[6]  = interpolateEdge(p[6], p[7], v[6], v[7], iso);
    if (edges &  128) edgeVerts[7]  = interpolateEdge(p[7], p[4], v[7], v[4], iso);
    if (edges &  256) edgeVerts[8]  = interpolateEdge(p[0], p[4], v[0], v[4], iso);
    if (edges &  512) edgeVerts[9]  = interpolateEdge(p[1], p[5], v[1], v[5], iso);
    if (edges & 1024) edgeVerts[10] = interpolateEdge(p[2], p[6], v[2], v[6], iso);
    if (edges & 2048) edgeVerts[11] = interpolateEdge(p[3], p[7], v[3], v[7], iso);

    // Compute normals from gradient at each edge intersection
    float3 edgeNormals[12];
    for (int e = 0; e < 12; e++) {
        if (edges & (1u << e)) {
            // Find closest grid point to approximate gradient
            float3 gp = (edgeVerts[e] - origin) / sp;
            uint gix = clamp(uint(gp.x + 0.5f), 0u, nx - 1u);
            uint giy = clamp(uint(gp.y + 0.5f), 0u, ny - 1u);
            uint giz = clamp(uint(gp.z + 0.5f), 0u, nz - 1u);
            edgeNormals[e] = computeGradient(scalarField, gix, giy, giz, nx, ny, nz);
        }
    }

    // Emit triangles
    for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
        int e0 = triTable[cubeIndex][i];
        int e1 = triTable[cubeIndex][i + 1];
        int e2 = triTable[cubeIndex][i + 2];

        // Allocate 3 vertices and 3 indices atomically
        uint vBase = atomic_fetch_add_explicit(vertexCounter, 3, memory_order_relaxed);
        uint iBase = atomic_fetch_add_explicit(indexCounter, 3, memory_order_relaxed);

        if (vBase + 3 > params.maxVertices || iBase + 3 > params.maxIndices) return;

        // Default color: light gray, will be overwritten by ESP coloring if desired
        float4 defaultColor = float4(0.82f, 0.84f, 0.86f, 0.85f);

        vertices[vBase + 0].position = edgeVerts[e0];
        vertices[vBase + 0].normal   = edgeNormals[e0];
        vertices[vBase + 0].color    = defaultColor;

        vertices[vBase + 1].position = edgeVerts[e1];
        vertices[vBase + 1].normal   = edgeNormals[e1];
        vertices[vBase + 1].color    = defaultColor;

        vertices[vBase + 2].position = edgeVerts[e2];
        vertices[vBase + 2].normal   = edgeNormals[e2];
        vertices[vBase + 2].color    = defaultColor;

        indices[iBase + 0] = vBase + 0;
        indices[iBase + 1] = vBase + 1;
        indices[iBase + 2] = vBase + 2;
    }
}

// ============================================================================
// MARK: - Electrostatic Potential Coloring
// ============================================================================

/// Compute electrostatic potential (ESP) coloring for surface vertices using
/// Coulomb's law with distance-dependent dielectric (Mehler-Solmajer model).
/// Each thread processes one vertex.
kernel void computeSurfaceESP(
    device SurfaceVertex       *vertices    [[buffer(0)]],
    constant SurfaceAtom       *atoms       [[buffer(1)]],
    constant SurfaceGridParams &params      [[buffer(2)]],
    device const uint          *vertexCount [[buffer(3)]],
    uint                        tid         [[thread_position_in_grid]])
{
    uint numVerts = vertexCount[0];
    if (tid >= numVerts) return;

    float3 vertPos = float3(vertices[tid].position);
    float esp = 0.0f;
    float cutoff = 20.0f;
    float cutoffSq = cutoff * cutoff;

    // Coulomb constant in kcal*Angstrom/(mol*e^2)
    const float kCoulomb = 332.06f;

    for (uint i = 0; i < params.numAtoms; i++) {
        float q = atoms[i].charge;
        if (abs(q) < 1e-6f) continue;

        float3 atomPos = float3(atoms[i].position);
        float3 diff = vertPos - atomPos;
        float distSq = dot(diff, diff);

        if (distSq > cutoffSq) continue;

        float dist = max(sqrt(distSq), 0.5f);

        // Distance-dependent dielectric (Mehler-Solmajer sigmoidal model)
        // eps(r) = A + B / (1 + k * exp(-lambda * B * r))
        // Parameters from Mehler & Solmajer 1991
        float A = -8.5525f;
        float B = 78.4f - A; // B = epsilon_solvent - A
        float k = 7.7839f;
        float lambda = 0.003627f;
        float eps = A + B / (1.0f + k * exp(-lambda * B * dist));
        eps = max(eps, 1.0f); // clamp to avoid division issues

        esp += kCoulomb * q / (eps * dist);
    }

    // Adaptive scale: map ESP onto [-1, 1] for coloring
    // Typical range is about +/-10 kcal/mol for druglike molecules
    float normalized = clamp(esp / 10.0f, -1.0f, 1.0f);

    float3 color;
    if (normalized < 0.0f) {
        // Negative potential (electron-rich): white -> red
        float t = -normalized;
        color = mix(float3(1.0f, 1.0f, 1.0f), float3(0.9f, 0.12f, 0.12f), t);
    } else {
        // Positive potential (electron-poor): white -> blue
        float t = normalized;
        color = mix(float3(1.0f, 1.0f, 1.0f), float3(0.12f, 0.22f, 0.92f), t);
    }

    vertices[tid].color = float4(color, 0.85f);
}

// ============================================================================
// MARK: - Hydrophobicity Surface Coloring
// ============================================================================

/// Color surface by hydrophobicity: brown/tan for hydrophobic (C, S), blue for
/// polar (N, O with charge), white for neutral. Each thread processes one vertex.
kernel void computeSurfaceHydrophobicity(
    device SurfaceVertex       *vertices    [[buffer(0)]],
    constant SurfaceAtom       *atoms       [[buffer(1)]],
    constant SurfaceGridParams &params      [[buffer(2)]],
    device const uint          *vertexCount [[buffer(3)]],
    uint                        tid         [[thread_position_in_grid]])
{
    uint numVerts = vertexCount[0];
    if (tid >= numVerts) return;

    float3 vertPos = float3(vertices[tid].position);

    // Find nearest atom
    float minDistSq = 1e20f;
    uint nearestIdx = 0;
    for (uint i = 0; i < params.numAtoms; i++) {
        float3 diff = vertPos - float3(atoms[i].position);
        float dSq = dot(diff, diff);
        if (dSq < minDistSq) {
            minDistSq = dSq;
            nearestIdx = i;
        }
    }

    uint anum = atoms[nearestIdx].atomicNum;
    float q = atoms[nearestIdx].charge;

    // Hydrophobicity classification
    // C(6), S(16) without large charge => hydrophobic
    // N(7), O(8) or any atom with |charge| > 0.3 => polar/hydrophilic
    // Others => intermediate
    float3 color;
    if ((anum == 6 || anum == 16) && abs(q) < 0.3f) {
        // Hydrophobic: golden brown
        color = float3(0.82f, 0.62f, 0.18f);
    } else if (anum == 7 || anum == 8 || abs(q) > 0.3f) {
        // Hydrophilic/polar: blue-teal
        color = float3(0.15f, 0.45f, 0.82f);
    } else if (anum == 9 || anum == 17 || anum == 35 || anum == 53) {
        // Halogens: slightly hydrophobic, pale green
        color = float3(0.45f, 0.72f, 0.35f);
    } else if (anum == 15) {
        // Phosphorus: orange
        color = float3(0.85f, 0.55f, 0.15f);
    } else {
        // Default: neutral gray-white
        color = float3(0.85f, 0.85f, 0.85f);
    }

    vertices[tid].color = float4(color, 0.85f);
}

// ============================================================================
// MARK: - Pharmacophore Surface Coloring
// ============================================================================

/// Color surface by pharmacophoric features:
///   H-bond donor (N-H, O-H): blue
///   H-bond acceptor (N, O lone pairs): red
///   Hydrophobic (C, halogen): yellow
///   Aromatic ring: purple
///   Charged positive: deep blue
///   Charged negative: deep red
kernel void computeSurfacePharmacophore(
    device SurfaceVertex       *vertices    [[buffer(0)]],
    constant SurfaceAtom       *atoms       [[buffer(1)]],
    constant SurfaceGridParams &params      [[buffer(2)]],
    device const uint          *vertexCount [[buffer(3)]],
    uint                        tid         [[thread_position_in_grid]])
{
    uint numVerts = vertexCount[0];
    if (tid >= numVerts) return;

    float3 vertPos = float3(vertices[tid].position);

    // Find nearest atom
    float minDistSq = 1e20f;
    uint nearestIdx = 0;
    for (uint i = 0; i < params.numAtoms; i++) {
        float3 diff = vertPos - float3(atoms[i].position);
        float dSq = dot(diff, diff);
        if (dSq < minDistSq) {
            minDistSq = dSq;
            nearestIdx = i;
        }
    }

    uint anum = atoms[nearestIdx].atomicNum;
    float q = atoms[nearestIdx].charge;
    uint aromatic = atoms[nearestIdx].isAromatic;

    float3 color;
    if (aromatic == 1) {
        // Aromatic: purple
        color = float3(0.62f, 0.28f, 0.82f);
    } else if (q > 0.3f && (anum == 7)) {
        // Positive charge (e.g., protonated amine): deep blue
        color = float3(0.10f, 0.25f, 0.90f);
    } else if (q < -0.3f && (anum == 8)) {
        // Negative charge (e.g., carboxylate): deep red
        color = float3(0.90f, 0.10f, 0.10f);
    } else if (anum == 7 && q > 0.0f) {
        // N-H donor: sky blue
        color = float3(0.30f, 0.60f, 0.95f);
    } else if (anum == 8 && q > 0.0f) {
        // O-H donor: sky blue
        color = float3(0.30f, 0.60f, 0.95f);
    } else if (anum == 7 || anum == 8) {
        // N/O acceptor: red-pink
        color = float3(0.92f, 0.35f, 0.35f);
    } else if (anum == 16) {
        // Sulfur: dark yellow
        color = float3(0.80f, 0.72f, 0.20f);
    } else if (anum == 9 || anum == 17 || anum == 35 || anum == 53) {
        // Halogens: green
        color = float3(0.35f, 0.75f, 0.30f);
    } else if (anum == 6) {
        // Carbon: hydrophobic yellow
        color = float3(0.92f, 0.82f, 0.30f);
    } else {
        // Default: light gray
        color = float3(0.80f, 0.80f, 0.80f);
    }

    vertices[tid].color = float4(color, 0.85f);
}

// Legacy alias using the old kernel name
kernel void computeESPColoring(
    device SurfaceVertex       *vertices    [[buffer(0)]],
    constant SurfaceAtom       *atoms       [[buffer(1)]],
    constant SurfaceGridParams &params      [[buffer(2)]],
    device const uint          *vertexCount [[buffer(3)]],
    uint                        tid         [[thread_position_in_grid]])
{
    // Delegate to the new implementation
    uint numVerts = vertexCount[0];
    if (tid >= numVerts) return;

    float3 vertPos = float3(vertices[tid].position);
    float esp = 0.0f;
    float cutoff = 20.0f;
    float cutoffSq = cutoff * cutoff;

    const float kCoulomb = 332.06f;

    for (uint i = 0; i < params.numAtoms; i++) {
        float q = atoms[i].charge;
        if (abs(q) < 1e-6f) continue;

        float3 atomPos = float3(atoms[i].position);
        float3 diff = vertPos - atomPos;
        float distSq = dot(diff, diff);
        if (distSq > cutoffSq) continue;

        float dist = max(sqrt(distSq), 0.5f);
        float A = -8.5525f;
        float B = 78.4f - A;
        float k = 7.7839f;
        float lambda = 0.003627f;
        float eps = A + B / (1.0f + k * exp(-lambda * B * dist));
        eps = max(eps, 1.0f);
        esp += kCoulomb * q / (eps * dist);
    }

    float normalized = clamp(esp / 10.0f, -1.0f, 1.0f);
    float3 color;
    if (normalized < 0.0f) {
        float t = -normalized;
        color = mix(float3(1.0f, 1.0f, 1.0f), float3(0.9f, 0.12f, 0.12f), t);
    } else {
        float t = normalized;
        color = mix(float3(1.0f, 1.0f, 1.0f), float3(0.12f, 0.22f, 0.92f), t);
    }
    vertices[tid].color = float4(color, 0.85f);
}
