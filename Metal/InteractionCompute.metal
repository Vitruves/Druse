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
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numLigandAtoms) return;

    InteractionAtomGPU ligAtom = ligAtoms[tid];
    float3 lp = ligPositions[tid];
    uint ligFlags = ligAtom.flags;
    int ligCharge = ligAtom.formalCharge;

    // Track whether this ligand atom found a strong (non-hydrophobic) interaction
    bool hasStrongInteraction = false;
    uint hydroCount = 0;
    const uint maxHydroPerAtom = 3;

    for (uint pi = 0; pi < params.numProteinAtoms; pi++) {
        InteractionAtomGPU protAtom = protAtoms[pi];
        float d = distance(lp, protAtom.position);
        if (d >= 6.0f) continue;

        uint protFlags = protAtom.flags;
        uint type = 0xFFFFFFFF; // sentinel: no interaction

        // ---- Metal coordination: < 2.8 Å, metal ↔ N/O/S ----
        if (d < 2.8f) {
            bool ligCoord  = (ligFlags  & (IDET_FLAG_N | IDET_FLAG_O | IDET_FLAG_S)) != 0;
            bool protMetal = (protFlags & IDET_FLAG_METAL) != 0;
            bool ligMetal  = (ligFlags  & IDET_FLAG_METAL) != 0;
            bool protCoord = (protFlags & (IDET_FLAG_N | IDET_FLAG_O | IDET_FLAG_S)) != 0;

            if ((protMetal && ligCoord) || (ligMetal && protCoord)) {
                type = 6; // metalCoord
            }
        }

        // ---- Salt bridge: < 4.0 Å, charged group ↔ charged group ----
        if (type == 0xFFFFFFFF && d < 4.0f) {
            bool protPositive = (protFlags & IDET_FLAG_POS_RES) != 0;
            bool protNegative = (protFlags & IDET_FLAG_NEG_RES) != 0;
            bool ligPositive  = ligCharge > 0;
            bool ligNegative  = ligCharge < 0;

            if ((protPositive && ligNegative) || (protNegative && ligPositive)) {
                type = 2; // saltBridge
            }
        }

        // ---- H-bond: 2.2-3.5 Å between donor/acceptor (N/O) ----
        if (type == 0xFFFFFFFF && d >= 2.2f && d <= 3.5f) {
            bool ligDA  = (ligFlags  & (IDET_FLAG_N | IDET_FLAG_O)) != 0;
            bool protDA = (protFlags & (IDET_FLAG_N | IDET_FLAG_O)) != 0;
            if (ligDA && protDA) {
                type = 0; // hbond
            }
        }

        // ---- Halogen bond: 2.5-3.5 Å, halogen ↔ N/O ----
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

        // ---- Hydrophobic: 3.3-4.5 Å, C/S ↔ C/S ----
        // Only if no strong interaction found for this ligand atom, max 3 per atom
        if (type == 0xFFFFFFFF && !hasStrongInteraction &&
            d >= 3.3f && d <= 4.5f && hydroCount < maxHydroPerAtom) {
            bool ligHydro  = (ligFlags  & (IDET_FLAG_C | IDET_FLAG_S)) != 0;
            bool protHydro = (protFlags & (IDET_FLAG_C | IDET_FLAG_S)) != 0;
            if (ligHydro && protHydro) {
                type = 1; // hydrophobic
                hydroCount++;
            }
        }

        // Atomically append to output buffer
        if (type != 0xFFFFFFFF) {
            uint idx = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
            if (idx < params.maxInteractions) {
                GPUInteraction inter;
                inter.ligandAtomIndex = tid;
                inter.proteinAtomIndex = pi;
                inter.type = type;
                inter.distance = d;
                inter.ligandPosition = lp;
                inter.proteinPosition = protAtom.position;
                output[idx] = inter;
            }
        }
    }
}
