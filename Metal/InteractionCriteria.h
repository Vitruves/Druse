// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#ifndef INTERACTION_CRITERIA_H
#define INTERACTION_CRITERIA_H

// ============================================================================
// MARK: - Interaction Geometry Criteria — SINGLE SOURCE OF TRUTH
// ============================================================================
//
// Every distance window, angular threshold and ramp width used by
//   * the Drusina scoring kernel      (Metal/DockingScoring.metal)
//   * the GPU interaction detector    (Metal/InteractionCompute.metal)
//   * the CPU interaction detector    (Docking/DockingEngine+Analysis.swift)
// lives here and nowhere else. Scorer and detector MUST agree: if they drift,
// the 3D interaction view contradicts the score breakdown.
//
// This header is visible to Metal (as `constant`) and to Swift (through
// Metal/Druse-Bridging-Header.h, as imported `let`s).
//
// Values are literature-standard starting points; each block cites its source.
// A separate measurement pass over the PDBbind refined set will replace the
// numbers — change them HERE ONLY and both scorer and detector follow.
//
// Naming: ...DistMin/Max are hard gates (outside = no interaction at all),
// ...DistOpt is the optimum, ...RampInner/Outer are the Verdonk-2003 linear
// ramp half-widths around the optimum (full score within Inner, zero beyond
// Outer). ...CosMin/CosMax are cosine thresholds, named for the cosine, so
// "CosMax = -0.766" means "angle >= 140 deg".
//
// References
//   Auffinger 2004   PNAS 101:16789      halogen bonds in biology
//   Baker 1984       Prog Biophys 44:97  H-bond geometry in proteins
//   Beno 2015        J Med Chem 58:4383  sulfur sigma-hole interactions
//   Bissantz 2010    J Med Chem 53:5061  medicinal chemist's interaction guide
//   Brandl 2001      J Mol Biol 307:357  C-H...pi in protein structures
//   Donald 2011      Proteins 79:898     salt bridges in proteins
//   Gallivan 1999    PNAS 96:9459        cation-pi in protein structures
//   Harder 2013      ChemMedChem 8:397   amide-pi stacking
//   McGaughey 1998   JBC 273:15458       pi-stacking geometry
//   Nishio 2011      PCCP 13:13873       CH/pi hydrogen bonds
//   Pascoe 2017      JACS 139:15160      chalcogen bond strength/geometry
//   Salentin 2015    NAR 43:W443         PLIP detection thresholds
//   Scholfield 2013  Protein Sci 22:139  halogen bonding in biology
//   Verdonk 2003     Proteins 52:609     GOLD block/ramp scoring functions
//   Zheng 2008       Nat Protoc 3:264    metal binding site geometry
// ============================================================================

#ifdef __METAL_VERSION__
#define IC_F constant float
#define IC_I constant int
#else
#define IC_F static const float
#define IC_I static const int
#endif

// ---- H-bond: heavy-atom donor...acceptor (N/O...N/O) --------------------
// Baker 1984, Bissantz 2010 Table 2 (CSD N...O median 2.9 A).
// Vina's own H-bond term is distance-only, so the detector is distance-only
// too; the Drusina directionality term adds the angular bonus on top and
// shares this distance window.
IC_F kICHBondDistMin        = 2.40f;   // closer than this is a clash, not an H-bond
IC_F kICHBondDistOpt        = 2.90f;
IC_F kICHBondDistMax        = 3.50f;
IC_F kICHBondRampInner      = 0.20f;
IC_F kICHBondRampOuter      = 0.60f;
// Donor angle X-D...A: 0 at 120 deg, full at 180 deg.
IC_F kICHBondCosDonorBad    = -0.50f;
IC_F kICHBondCosDonorGood   = -1.00f;
// Acceptor angle D...A-Y: 0 at 70 deg, full at 100 deg (sp2 lone pair).
IC_F kICHBondCosAccBad      = 0.34f;
IC_F kICHBondCosAccGood     = -0.17f;
// Minimum combined angular factor before the directionality bonus applies.
IC_F kICHBondAngleFactorMin = 0.25f;

// ---- Hydrophobic contact: C/S ... C/S ----------------------------------
// PLIP default (Salentin 2015): hydrophobic carbon contact <= 4.0 A. Vina's
// hydrophobic term fades to zero at surface distance 1.5 A (~5.3 A for C-C),
// so this window is the strong core of what Vina rewards, not its tail.
IC_F kICHydrophobicDistMin  = 3.40f;
IC_F kICHydrophobicDistMax  = 4.00f;
IC_I kICHydrophobicMaxPerAtom = 2;

// ---- Salt bridge: charged group ... charged group -----------------------
// Bissantz 2010 Table 2 (CSD median N...O 2.79 A); Donald 2011 for per-group
// (not per-atom) pairing; Bissantz 2010 for burial weighting of exposed pairs.
IC_F kICSaltBridgeDistOpt   = 2.80f;
IC_F kICSaltBridgeDistMax   = 4.00f;
IC_F kICSaltBridgeRampInner = 0.30f;
IC_F kICSaltBridgeRampOuter = 1.20f;
// Partial-charge fallback when no formal charge is assigned (neutral SMILES):
// |q| above this counts as ionic, at reduced confidence.
IC_F kICSaltBridgeChargeThreshold = 0.35f;
IC_F kICSaltBridgePartialScale    = 0.50f;

// ---- pi-pi stacking: aromatic ring ... aromatic ring -------------------
// Bissantz 2010 Table 2 / McGaughey 1998: parallel-displaced 3.4-3.6 A,
// T-shaped 4.5-5.5 A. Lateral offset cap from PLIP (Salentin 2015).
IC_F kICPiStackDistMin      = 3.20f;
IC_F kICPiStackFaceDistOpt  = 3.50f;
IC_F kICPiStackFaceDistMax  = 4.50f;
IC_F kICPiStackEdgeDistOpt  = 4.80f;
IC_F kICPiStackEdgeDistMax  = 5.80f;
IC_F kICPiStackFaceCosMin   = 0.80f;   // interplanar angle <= 37 deg
IC_F kICPiStackEdgeCosMax   = 0.40f;   // interplanar angle >= 66 deg
IC_F kICPiStackOffsetMax    = 2.00f;   // lateral centroid offset, face-to-face only
IC_F kICPiStackFaceRampInner = 0.30f;
IC_F kICPiStackFaceRampOuter = 1.00f;
IC_F kICPiStackEdgeRampInner = 0.40f;
IC_F kICPiStackEdgeRampOuter = 1.00f;
IC_F kICPiStackEdgeScale     = 0.60f;  // T-shaped is weaker than face-to-face

// ---- pi-cation: aromatic ring ... cation --------------------------------
// Gallivan 1999 / Bissantz 2010: cation 3.4-4.0 A above the ring plane,
// approach along the ring normal. Offset cap from PLIP.
IC_F kICPiCationDistOpt     = 3.80f;
IC_F kICPiCationDistMax     = 5.00f;   // = DistOpt + RampOuter, gate and ramp agree
IC_F kICPiCationCosMin      = 0.50f;   // within 60 deg of the ring normal
IC_F kICPiCationOffsetMax   = 2.00f;
IC_F kICPiCationRampInner   = 0.40f;
IC_F kICPiCationRampOuter   = 1.20f;

// ---- Halogen bond: C-X ... O/N (X = Cl, Br, I) -------------------------
// Auffinger 2004 / Scholfield 2013 CSD statistics for the optima; PLIP donor
// angle 165 +/- 30 deg gives the 140 deg sigma-hole gate. Organic fluorine
// has no usable sigma-hole (Auffinger 2004) and is excluded on both paths.
IC_F kICHalogenDistMin      = 2.50f;
IC_F kICHalogenDistMax      = 4.00f;
IC_F kICHalogenDistOptCl    = 3.27f;
IC_F kICHalogenDistOptBr    = 3.24f;
IC_F kICHalogenDistOptI     = 3.17f;
IC_F kICHalogenCosMax       = -0.766f; // C-X...A >= 140 deg
IC_F kICHalogenRampInner    = 0.30f;
IC_F kICHalogenRampOuter    = 0.80f;

// ---- Chalcogen bond: C-S ... O/N ---------------------------------------
// Beno 2015 (S...O 3.0-3.6 A), Pascoe 2017 (sigma-hole usable from 140 deg).
IC_F kICChalcogenDistMin    = 2.80f;
IC_F kICChalcogenDistOpt    = 3.30f;
IC_F kICChalcogenDistMax    = 4.00f;
IC_F kICChalcogenCosMax     = -0.766f; // C-S...A >= 140 deg
IC_F kICChalcogenRampInner  = 0.30f;
IC_F kICChalcogenRampOuter  = 0.80f;

// ---- Metal coordination: metal ... N/O/S -------------------------------
// Zheng 2008 / CSD: Zn-O 2.05, Zn-N 2.1, Mg-O 2.07, Ca-O 2.4 A. PLIP uses a
// 3.0 A cap. DistMax is the detection shell; ScoreCutoff is the outer radius
// of the scoring funnel, which stays wider so the GA has a gradient to follow.
IC_F kICMetalDistMin        = 1.70f;
IC_F kICMetalDistOpt        = 2.40f;
IC_F kICMetalDistMax        = 2.80f;
IC_F kICMetalScoreCutoff    = 4.50f;
IC_F kICMetalRampInner      = 0.40f;
IC_F kICMetalRampOuter      = 1.60f;

// ---- CH-pi: aliphatic C-H ... aromatic ring ----------------------------
// Nishio 2011 / Brandl 2001 PDB survey: C...ring-centroid 3.4-4.5 A with the
// C within ~45 deg of the ring normal. Scored without explicit hydrogens, so
// the heavy-atom carbon stands in for the C-H vector.
IC_F kICCHPiDistMin         = 3.40f;
IC_F kICCHPiDistOpt         = 3.80f;
IC_F kICCHPiDistMax         = 4.50f;
IC_F kICCHPiCosMin          = 0.70f;   // within 45 deg of the ring normal
IC_F kICCHPiOffsetMax       = 2.00f;
IC_F kICCHPiRampInner       = 0.20f;
IC_F kICCHPiRampOuter       = 0.70f;

// ---- Amide-pi: backbone/sidechain amide plane ... aromatic ring --------
// Harder 2013 / Bissantz 2010: amide-aryl interplanar separation 3.2-3.7 A,
// parallel stacking (interplanar angle <= 30-37 deg). The looser "tilted"
// regime is not supported by the surveys and is not scored.
IC_F kICAmidePiDistMin      = 3.20f;
IC_F kICAmidePiDistOpt      = 3.60f;
IC_F kICAmidePiDistMax      = 4.50f;
IC_F kICAmidePiCosMin       = 0.80f;   // interplanar angle <= 37 deg
IC_F kICAmidePiOffsetMax    = 2.00f;
IC_F kICAmidePiRampInner    = 0.30f;
IC_F kICAmidePiRampOuter    = 0.90f;

// ---- Burial / desolvation ----------------------------------------------
// Protein heavy atoms within this radius of a ligand atom count as contacts.
IC_F kICBurialContactRadius = 5.00f;
// A polar ligand atom is "deeply buried" from this contact count upward, and
// fully buried kContactsFull contacts later (HYDE, Schneider 2013).
IC_I kICBurialPolarMinContacts = 12;
IC_I kICBurialPolarFullContacts = 25;
// A hydrophobic ligand atom is solvent-exposed below this many close contacts.
IC_F kICExposureContactRadius = 4.50f;
IC_I kICExposureMinContacts   = 3;
// Screened Coulomb is attenuated for solvent-exposed ligand atoms: an atom
// with no protein contact is screened by water, not by the protein interior
// (Bissantz 2010 on exposed ionic pairs).
IC_F kICCoulombBurialMinScale = 0.20f;

// ---- Global bounds -----------------------------------------------------
// Largest centre-to-centre reach of any term above (pi-pi T-shaped, 5.8 A),
// plus slack. Used to cull protein features that no ligand atom can reach.
IC_F kICMaxInteractionReach = 6.50f;
// The Drusina correction may not exceed this fraction of the favourable Vina
// energy. Vina carries the steric/desolvation landscape that locates the
// pocket; the extended terms are a refinement on top of it and must never
// outvote it (see Benchmark/reports/drusina_regression_analysis.md, RC1).
IC_F kICDrusinaMaxVinaFraction = 0.50f;

#endif /* INTERACTION_CRITERIA_H */
