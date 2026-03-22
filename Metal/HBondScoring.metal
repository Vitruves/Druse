#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

// ============================================================================
// MARK: - H-Bond Network Scoring Kernel
// ============================================================================
//
// Scores candidate atom positions (from moveable group states) against the
// fixed protein environment. One thread per candidate atom.
//
// Implements a simplified Reduce-style scoring:
//   - H-bond:  favorable when donor H approaches acceptor within distance range
//   - Bump:    penalty for VDW overlap (steric clash)
//   - Contact: weak favorable for van der Waals contact (Gaussian decay)
//
// Reference: Word et al., J Mol Biol 1999
// ============================================================================

kernel void scoreHBondCandidates(
    device const HBondCandidateAtom *candidates [[buffer(0)]],
    device const HBondEnvAtom *envAtoms         [[buffer(1)]],
    device HBondAtomScore *scores               [[buffer(2)]],
    constant HBondScoringParams &params         [[buffer(3)]],
    device const uint32_t *excludeMask          [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.numCandidates) return;

    HBondCandidateAtom candidate = candidates[tid];
    float3 cPos = candidate.position;
    float cRad = candidate.vdwRadius;
    uint cFlags = candidate.flags;
    int cCharge = candidate.formalCharge;

    bool cIsDonor    = (cFlags & HBNET_FLAG_DONOR) != 0;
    bool cIsAcceptor = (cFlags & HBNET_FLAG_ACCEPTOR) != 0;
    bool cIsHydrogen = (cFlags & HBNET_FLAG_HYDROGEN) != 0;
    bool cIsCharged  = (cFlags & HBNET_FLAG_CHARGED) != 0;

    float bumpAccum = 0.0f;
    float hbondAccum = 0.0f;
    float contactAccum = 0.0f;
    bool hasBadBump = false;

    // Exclude mask: one bit per env atom, packed as uint32_t array
    // Bit j of excludeMask[tid * maskStride + j/32] indicates env atom j is excluded
    uint maskStride = (params.numEnvAtoms + 31u) / 32u;

    for (uint ei = 0; ei < params.numEnvAtoms; ei++) {
        // Check exclusion mask (bonded atoms within same residue)
        uint maskWord = excludeMask[tid * maskStride + ei / 32u];
        if ((maskWord >> (ei & 31u)) & 1u) continue;

        HBondEnvAtom envAtom = envAtoms[ei];
        float3 ePos = envAtom.position;
        float eRad = envAtom.vdwRadius;
        uint eFlags = envAtom.flags;
        int eCharge = envAtom.formalCharge;

        float d = distance(cPos, ePos);
        float sumRadii = cRad + eRad;

        // Skip atoms too far for any interaction
        if (d > sumRadii + 1.0f) continue;

        // Gap: positive = separated, negative = overlap
        float gap = d - sumRadii;

        // Determine if this is an H-bond interaction
        bool eIsDonor    = (eFlags & HBNET_FLAG_DONOR) != 0;
        bool eIsAcceptor = (eFlags & HBNET_FLAG_ACCEPTOR) != 0;
        bool eIsCharged  = (eFlags & HBNET_FLAG_CHARGED) != 0;

        bool isaHB = false;
        if ((cIsDonor && eIsAcceptor) || (cIsAcceptor && eIsDonor) ||
            (cIsHydrogen && eIsAcceptor) || (eFlags & HBNET_FLAG_HYDROGEN && (cIsAcceptor || cIsDonor))) {
            // Check charge compatibility: don't count H-bond if same-sign charges
            bool sameSign = (cCharge > 0 && eCharge > 0) || (cCharge < 0 && eCharge < 0);
            if (!sameSign) {
                isaHB = true;
            }
        }

        // Determine H-bond minimum distance threshold
        bool bothCharged = cIsCharged && eIsCharged && (cCharge * eCharge < 0);
        float hbMindist = bothCharged ? params.minChargedHBGap : params.minRegHBGap;

        if (gap > 0.0f) {
            // Contact: Gaussian decay for favorable VDW contact
            float scaledGap = gap / params.gapScale;
            float contactScore = exp(-scaledGap * scaledGap);
            contactAccum += contactScore;
        } else if (isaHB) {
            bool tooClose = (gap < -hbMindist);
            if (tooClose) {
                // Too-close H-bond becomes a bump
                float adjustedGap = gap + hbMindist;
                float overlap = -0.5f * adjustedGap;
                bumpAccum -= params.bumpWeight * overlap;
                if (-adjustedGap >= params.badBumpGapCut) {
                    hasBadBump = true;
                }
            } else {
                // Good H-bond: reward proportional to overlap
                float overlap = -0.5f * gap;
                hbondAccum += params.hbondWeight * overlap;
            }
        } else {
            // Non-H-bond clash: steric penalty
            float overlap = -0.5f * gap;
            bumpAccum -= params.bumpWeight * overlap;
            if (-gap >= params.badBumpGapCut) {
                hasBadBump = true;
            }
        }
    }

    // Write per-atom scores
    HBondAtomScore result;
    result.totalScore = bumpAccum + hbondAccum + contactAccum;
    result.bumpScore = bumpAccum;
    result.hbondScore = hbondAccum;
    result.contactScore = contactAccum;
    result.hasBadBump = hasBadBump ? 1u : 0u;
    result._pad0 = 0;
    result._pad1 = 0;
    result._pad2 = 0;
    scores[tid] = result;
}
