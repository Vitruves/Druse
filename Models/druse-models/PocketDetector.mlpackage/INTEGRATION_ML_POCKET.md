# PocketDetector v3 -- 3D U-Net Integration Guide (macOS/iOS)

## Architecture

Voxelize protein atoms into a 3D grid, run a 3D U-Net, get per-voxel pocket
probabilities. All ops are standard Conv3D — natively supported by CoreML/Metal.

```
  Swift/Metal side                CoreML model                    Output
 ___________________    ____________________________    _____________________
|                   |  |                            |  |                     |
| 1. Parse PDB      |  |  3D U-Net (5.7M params)    |  | pocket_probability  |
| 2. Center atoms    |  |                            |  |  [1, 1, 64, 64, 64] |
| 3. Bin atoms into  |->|  Encoder:                  |->|                     |
|    64³ voxel grid  |  |    Conv3D(10→32) + pool     |  | Each voxel: 0.0-1.0 |
|    with 10 feature |  |    Conv3D(32→64) + pool     |  | 0.0 = not pocket    |
|    channels        |  |    Conv3D(64→128) + pool    |  | 1.0 = pocket        |
|                   |  |    Conv3D(128→256)           |  |                     |
| voxel_grid         |  |  Decoder (with skip conn):  |  | Threshold from      |
|  [1, 10, 64,64,64] |  |    Up + Conv3D → 128        |  | model metadata      |
|___________________|  |    Up + Conv3D → 64          |  |_____________________|
                       |    Up + Conv3D → 32          |
                       |    Conv3D(32→1) + sigmoid    |
                       |____________________________|
```

---

## CoreML Model

**Input**: `voxel_grid` — `Float32 [1, 10, 64, 64, 64]`

**Output**: `pocket_probability` — `Float32 [1, 1, 64, 64, 64]`

**Metadata** (read from model):
- `pocket_threshold`: optimal threshold from validation (e.g. `0.000511`)
- `grid_size`: `64`
- `resolution`: `1.0` (Angstroms per voxel)
- `n_channels`: `10`

---

## Voxel Grid Input Channels

Each voxel accumulates features from atoms that fall into it. One atom = one voxel
(nearest-neighbor binning at 1.0A resolution).

| Channel | Name | Values | Description |
|---------|------|--------|-------------|
| 0 | all_atom | count | Heavy atom count in this voxel |
| 1 | carbon | count | Carbon atoms |
| 2 | nitrogen | count | Nitrogen atoms |
| 3 | oxygen | count | Oxygen atoms |
| 4 | sulfur | count | Sulfur/phosphorus/other atoms |
| 5 | hydrophobicity | float | Kyte-Doolittle / 4.5 (normalized to ~[-1,1]) |
| 6 | charge | float | Partial charge at pH 7.4 (see table below) |
| 7 | aromatic | 0/1 | Aromatic ring atom (PHE, TYR, TRP, HIS) |
| 8 | hbond_donor | 0/1 | Backbone N (not PRO) or side-chain donor |
| 9 | hbond_acceptor | 0/1 | Backbone O or side-chain acceptor |

### Hydrophobicity values (Kyte-Doolittle, divide by 4.5)

```
ILE: 4.5  VAL: 4.2  LEU: 3.8  PHE: 2.8  CYS: 2.5  MET: 1.9  ALA: 1.8
GLY:-0.4  THR:-0.7  SER:-0.8  TRP:-0.9  TYR:-1.3  PRO:-1.6  HIS:-3.2
ASN:-3.5  ASP:-3.5  GLN:-3.5  GLU:-3.5  LYS:-3.9  ARG:-4.5
```

### Partial charge values (pH 7.4)

```
ASP-OD1: -0.5    ASP-OD2: -0.5
GLU-OE1: -0.5    GLU-OE2: -0.5
LYS-NZ:  +1.0
ARG-NH1: +0.33   ARG-NH2: +0.33   ARG-NE: +0.33
HIS-ND1: +0.25   HIS-NE2: +0.25
All other atoms: 0.0
```

---

## Voxelization (Swift/Metal pseudocode)

```swift
let gridSize = 64
let resolution: Float = 1.0  // 1 Angstrom per voxel

func voxelizeProtein(atoms: [Atom]) -> MLMultiArray {
    // 1. Compute grid center (protein centroid)
    let center = atoms.map(\.position).mean()
    let half = Float(gridSize) * resolution / 2.0

    // 2. Create empty grid [1, 10, 64, 64, 64]
    var grid = MLMultiArray(shape: [1, 10, 64, 64, 64], dataType: .float32)

    // 3. Bin each atom into its nearest voxel
    for atom in atoms {
        guard atom.element != "H" else { continue }

        let vx = Int((atom.x - center.x + half) / resolution)
        let vy = Int((atom.y - center.y + half) / resolution)
        let vz = Int((atom.z - center.z + half) / resolution)

        guard (0..<gridSize).contains(vx),
              (0..<gridSize).contains(vy),
              (0..<gridSize).contains(vz) else { continue }

        // Channel 0: all-atom density
        grid[[0, 0, vx, vy, vz]] += 1.0

        // Channels 1-4: element type
        switch atom.element {
        case "C": grid[[0, 1, vx, vy, vz]] += 1.0
        case "N": grid[[0, 2, vx, vy, vz]] += 1.0
        case "O": grid[[0, 3, vx, vy, vz]] += 1.0
        default:  grid[[0, 4, vx, vy, vz]] += 1.0  // S, P, etc.
        }

        // Channel 5: hydrophobicity (from residue)
        grid[[0, 5, vx, vy, vz]] += hydrophobicity[atom.residueName] / 4.5

        // Channel 6: partial charge (from residue + atom name)
        grid[[0, 6, vx, vy, vz]] += partialCharge(atom)

        // Channel 7: aromatic
        if isAromatic(atom) { grid[[0, 7, vx, vy, vz]] += 1.0 }

        // Channel 8: H-bond donor
        if isHBondDonor(atom) { grid[[0, 8, vx, vy, vz]] += 1.0 }

        // Channel 9: H-bond acceptor
        if isHBondAcceptor(atom) { grid[[0, 9, vx, vy, vz]] += 1.0 }
    }

    return grid
}
```

This can also be implemented as a **Metal compute shader** for GPU acceleration —
it's just a scatter operation (one thread per atom, atomic add to grid).

---

## Full Inference Pipeline

```swift
// 1. Parse protein
let atoms = parsePDB(url: pdbFileURL)

// 2. Compute grid center (needed to convert voxels back to world coords)
let center = atoms.map(\.position).mean()

// 3. Voxelize
let voxelGrid = voxelizeProtein(atoms: atoms)

// 4. Run CoreML model
let model = try PocketDetector(configuration: .init())
let output = model.prediction(voxel_grid: voxelGrid)
let probGrid = output.pocket_probability  // [1, 1, 64, 64, 64]

// 5. Read threshold from model metadata
let threshold = Float(model.model.modelDescription
    .metadata[.init(rawValue: "pocket_threshold")] as? String ?? "0.01") ?? 0.01

// 6. Extract pocket voxels → world coordinates
let resolution: Float = 1.0
let half = Float(64) * resolution / 2.0
var pocketPoints: [(SIMD3<Float>, Float)] = []

for x in 0..<64 {
    for y in 0..<64 {
        for z in 0..<64 {
            let prob = probGrid[[0, 0, x, y, z]].floatValue
            if prob > threshold {
                let worldPos = SIMD3<Float>(
                    Float(x) * resolution - half + center.x,
                    Float(y) * resolution - half + center.y,
                    Float(z) * resolution - half + center.z
                )
                pocketPoints.append((worldPos, prob))
            }
        }
    }
}

// 7. Cluster into binding sites (DBSCAN or connected components)
let sites = clusterPockets(pocketPoints, eps: 4.0, minPts: 3)

// 8. Rank by mean probability
let ranked = sites.sorted { $0.meanProbability > $1.meanProbability }
```

---

## Pocket Clustering

After thresholding, cluster positive voxels into discrete binding sites.

**Connected components** (simplest, works on 3D grid):
```swift
// Flood-fill on the thresholded 3D binary grid
// Each connected region = one predicted binding site
// Use 6-connectivity (face neighbors) or 26-connectivity (all neighbors)
```

**Or DBSCAN** on the extracted world-coordinate points:
- `eps = 4.0` Angstroms
- `minPts = 3` voxels

**Per-site scoring**:
- `meanProbability`: average prob of all voxels in the site
- `center`: centroid of site voxels (in Angstroms, for docking)
- `volume`: number of voxels * resolution³ (approximate cavity volume in A³)

---

## Performance

| Step | Time (M1 Pro) | Notes |
|------|---------------|-------|
| Parse PDB | ~5ms | Standard PDB parser |
| Voxelize | ~1ms | 1 loop over atoms, no KNN/surface/KDTree |
| CoreML inference | ~20ms | Conv3D on Metal GPU |
| Extract + cluster | ~2ms | Threshold + flood fill on 64³ grid |
| **Total** | **~28ms** | |

Model size: ~22 MB (.mlpackage), 5.7M parameters.

---

## Retraining / Re-export

```bash
# Train (no precompute needed — voxelization is on-the-fly)
python train_pocket_detector.py --data_dir data/ --epochs 100 --batch_size 16 --workers 16

# Export to CoreML
python export_coreml.py --pocket_detector checkpoints/pocket_detector_best.pt \
    --output_dir /path/to/Druse/Models/druse-models/
```

---

## Troubleshooting

**No pockets found**: Check `pocket_threshold` from model metadata — it may be
very small (e.g. 0.0005). The model outputs low absolute probabilities but the
relative ranking is correct. Try a fixed threshold of 0.01 if metadata threshold
gives too many/few results.

**Wrong pocket locations**: Verify the `center` used for voxelization matches the
center used to convert voxel coords back to world coords. This must be the protein
centroid (mean of all heavy atom positions).

**Large proteins (>64A)**: The 64³ grid at 1A resolution covers 64A. Atoms outside
this range are clipped. For very large proteins, consider centering on the region
of interest rather than the full protein centroid.
