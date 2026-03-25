#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// ============================================================================
// MARK: - GPU Interaction Detection
// ============================================================================
//
// One thread per ligand atom. Each thread loops over all protein atoms,
// applies distance cutoffs and element-flag checks, and atomically appends
// detected interactions to an output buffer.
//
// Handles: metal coordination, salt bridges, H-bonds, halogen bonds,
//          hydrophobic contacts.
// CPU handles: π-π stacking, π-cation (require ring geometry).
// ============================================================================

kernel void detectInteractions(
    device const InteractionAtomGPU *protAtoms [[buffer(0)]],
    device const InteractionAtomGPU *ligAtoms  [[buffer(1)]],
    device const float3 *ligPositions          [[buffer(2)]],
    device GPUInteraction *output              [[buffer(3)]],
    device atomic_uint *counter                [[buffer(4)]],
    constant InteractionDetectParams &params   [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    // Maximum interactions a single ligand atom can produce.
    // 6 strong (metal/salt/hbond/halogen) + 2 hydrophobic = 8 typical max,
    // but protein atoms can have multiple contacts; 24 is a safe upper bound.
    const uint MAX_LOCAL = 24;
    GPUInteraction localInteractions[MAX_LOCAL];
    uint myCount = 0;

    if (tid < params.numLigandAtoms) {
        InteractionAtomGPU ligAtom = ligAtoms[tid];
        float3 lp = ligPositions[tid];
        uint ligFlags = ligAtom.flags;
        int ligCharge = ligAtom.formalCharge;

        // Track whether this ligand atom found a strong (non-hydrophobic) interaction
        bool hasStrongInteraction = false;
        uint hydroCount = 0;
        const uint maxHydroPerAtom = 2;

        for (uint pi = 0; pi < params.numProteinAtoms; pi++) {
            InteractionAtomGPU protAtom = protAtoms[pi];
            float d = distance(lp, protAtom.position);
            if (d >= 6.0f) continue;

            uint protFlags = protAtom.flags;
            uint type = 0xFFFFFFFF; // sentinel: no interaction

            // ---- Metal coordination: < 2.8 Angstrom, metal <-> N/O/S ----
            if (d < 2.8f) {
                bool ligCoord  = (ligFlags  & (IDET_FLAG_N | IDET_FLAG_O | IDET_FLAG_S)) != 0;
                bool protMetal = (protFlags & IDET_FLAG_METAL) != 0;
                bool ligMetal  = (ligFlags  & IDET_FLAG_METAL) != 0;
                bool protCoord = (protFlags & (IDET_FLAG_N | IDET_FLAG_O | IDET_FLAG_S)) != 0;

                if ((protMetal && ligCoord) || (ligMetal && protCoord)) {
                    type = 6; // metalCoord
                }
            }

            // ---- Salt bridge: < 4.0 Angstrom, charged group <-> charged group ----
            if (type == 0xFFFFFFFF && d < 4.0f) {
                bool protPositive = (protFlags & IDET_FLAG_POS_RES) != 0;
                bool protNegative = (protFlags & IDET_FLAG_NEG_RES) != 0;
                bool ligPositive  = ligCharge > 0;
                bool ligNegative  = ligCharge < 0;

                if ((protPositive && ligNegative) || (protNegative && ligPositive)) {
                    type = 2; // saltBridge
                }
            }

            // ---- H-bond: 2.2-3.5 Angstrom between donor/acceptor (N/O) ----
            if (type == 0xFFFFFFFF && d >= 2.2f && d <= 3.5f) {
                bool ligDA  = (ligFlags  & (IDET_FLAG_N | IDET_FLAG_O)) != 0;
                bool protDA = (protFlags & (IDET_FLAG_N | IDET_FLAG_O)) != 0;
                if (ligDA && protDA) {
                    type = 0; // hbond
                }
            }

            // ---- Halogen bond: 2.5-3.5 Angstrom, halogen <-> N/O ----
            if (type == 0xFFFFFFFF && d >= 2.5f && d <= 3.5f) {
                bool halogen  = (ligFlags  & IDET_FLAG_HALOGEN) != 0;
                bool acceptor = (protFlags & (IDET_FLAG_N | IDET_FLAG_O)) != 0;
                if (halogen && acceptor) {
                    type = 5; // halogen
                }
            }

            // Mark strong interactions (anything except hydrophobic)
            if (type != 0xFFFFFFFF) {
                hasStrongInteraction = true;
            }

            // ---- Hydrophobic: 3.4-4.0 Angstrom, C/S <-> C/S ----
            // Allow hydrophobic contacts even if this atom has a strong interaction elsewhere, max 2 per atom
            if (type == 0xFFFFFFFF &&
                d >= 3.4f && d <= 4.0f && hydroCount < maxHydroPerAtom) {
                bool ligHydro  = (ligFlags  & (IDET_FLAG_C | IDET_FLAG_S)) != 0;
                bool protHydro = (protFlags & (IDET_FLAG_C | IDET_FLAG_S)) != 0;
                if (ligHydro && protHydro) {
                    type = 1; // hydrophobic
                    hydroCount++;
                }
            }

            // Store to thread-local buffer instead of global atomic append
            if (type != 0xFFFFFFFF && myCount < MAX_LOCAL) {
                GPUInteraction inter;
                inter.ligandAtomIndex = tid;
                inter.proteinAtomIndex = pi;
                inter.type = type;
                inter.distance = d;
                inter.ligandPosition = lp;
                inter.proteinPosition = protAtom.position;
                localInteractions[myCount] = inter;
                myCount++;
            }
        }
    }

    // ---- Phase 2: Per-threadgroup reduction + single global atomic ----

    // Each thread stores its count into threadgroup memory
    threadgroup uint localCounts[256];
    localCounts[lid] = myCount;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 computes total for the group and allocates a contiguous block
    // groupOffset is written by thread 0; threadgroup_barrier ensures visibility to all threads.
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wsometimes-uninitialized"
    threadgroup uint groupOffset;
    threadgroup uint prefixSums[256];
    if (lid == 0) {
        uint running = 0;
        for (uint i = 0; i < tgSize; i++) {
            prefixSums[i] = running;
            running += localCounts[i];
        }
        // Single atomic allocation for the entire threadgroup
        if (running > 0) {
            groupOffset = atomic_fetch_add_explicit(counter, running, memory_order_relaxed);
        } else {
            groupOffset = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 3: Write interactions at the computed offset (no atomics) ----
    uint myOffset = groupOffset + prefixSums[lid];
    #pragma clang diagnostic pop
    for (uint i = 0; i < myCount; i++) {
        uint writeIdx = myOffset + i;
        if (writeIdx < params.maxInteractions) {
            output[writeIdx] = localInteractions[i];
        }
    }
}
