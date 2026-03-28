// FASPRSearch.swift
// Translated from FASPR C++ (Search.h / Search.cpp)
// Original: Copyright (c) 2020 Xiaoqiang Huang, MIT License
//
// Dead-End Elimination (DEE) followed by tree-decomposition exact search
// for the Global Minimum Energy Conformation (GMEC) of side-chain rotamers.

import Foundation

// MARK: - Constants

private let DEE_THRESHOLD: Float  = 0.0
private let EPAIR_CUT: Float      = 2.0
private let TREEWIDTH_CUT: Int    = 5
private let DEAD_ENERGY: Float    = 999.0
private let DEAD_MARK: Float      = 1000.0

// MARK: - Supporting Types

typealias AdjGraph = [Int: Set<Int>]

enum BagType { case root, inner, leaf, none }

struct TreeBag {
    var left: Set<Int>          // vertices inherited from parent
    var right: Set<Int>         // new vertices in this bag
    var total: Set<Int>         // left ∪ right
    var parentIdx: Int          // -1 for root
    var childIndices: Set<Int>  // indices of child bags
    var type: BagType

    // Flattened site / rotamer arrays used during DP
    var lsites: [Int] = []
    var rsites: [Int] = []
    var tsites: [Int] = []
    var lrots: [[Int]] = []
    var rrots: [[Int]] = []

    // DP tables: Etcom[lcomIdx][rcomIdx], Rtcom[lcomIdx] = best total rotamer combo
    var Etcom: [[Float]] = []
    var Rtcom: [[Int]] = []

    // Indices mapping child.lsites -> parent.tsites positions
    var indices: [Int] = []

    var deployFlag: Bool = false

    // GMEC backtrack
    var EGMEC: Float = 1e8
    var RLGMEC: [Int] = []
    var RRGMEC: [Int] = []

    var childCounter: Int = 0

    init() {
        left = []
        right = []
        total = []
        parentIdx = -1
        childIndices = []
        type = .leaf
    }
}

// MARK: - FASPRSearch

enum FASPRSearch {

    // MARK: Main Entry Point

    /// Run FASPR search: DEE elimination then tree-decomposition exact GMEC.
    /// - Parameters:
    ///   - nrots:      rotamer count per site (mutable copy used internally)
    ///   - eTableSelf: [site][rot] self-energy
    ///   - eTablePair: [siteI][siteJ] -> 2D array [rotI][rotJ]; absent key = no interaction
    ///   - conMap:      neighbor list per site (used to seed initial pos list)
    /// - Returns: bestrot[site], rotamer index chosen (-1 if site was never multi-rot)
    static func search(
        nrots nrotsIn: [Int],
        eTableSelf eTableSelfIn: [[Float]],
        eTablePair eTablePairIn: [Int: [Int: [[Float]]]],
        conMap: [[Int]]
    ) -> [Int] {
        let nres = nrotsIn.count
        var nrots = nrotsIn
        var eTableSelf = eTableSelfIn
        var eTablePair = eTablePairIn
        var bestrot = [Int](repeating: -1, count: nres)

        // Build initial pos list: sites with >1 rotamer
        var pos = [Int]()
        for i in 0..<nres {
            if nrots[i] > 1 { pos.append(i) }
        }

        // --- DEE Search (6 alternating rounds of Goldstein / Split) ---
        let deeHasUnfixed = deeSearch(
            pos: &pos,
            nres: nres,
            nrots: &nrots,
            eTableSelf: &eTableSelf,
            eTablePair: &eTablePair,
            bestrot: &bestrot
        )

        guard deeHasUnfixed else {
            return bestrot
        }

        // Collect unfixed residues after DEE
        var unfixres = [Int]()
        for i in 0..<nres {
            if nrots[i] > 1 { unfixres.append(i) }
        }

        // --- Graph construction + tree decomposition loop ---
        var hardmode = false
        var threshold: Float = 0.5

        while true {
            let nunfix = unfixres.count
            var adjMatrix = [[Int]](repeating: [Int](repeating: 0, count: nunfix), count: nunfix)
            var flagMatrix = [[Int]](repeating: [Int](repeating: 0, count: nunfix), count: nunfix)

            constructAdjMatrix(
                nunfix: nunfix,
                unfixres: unfixres,
                nrots: &nrots,
                eTableSelf: eTableSelf,
                eTablePair: &eTablePair,
                bestrot: &bestrot,
                adjMatrix: &adjMatrix
            )

            if hardmode {
                hardmode = false
                graphEdgeDecomposition(
                    adjMatrix: &adjMatrix,
                    threshold: threshold,
                    unfixres: unfixres,
                    nrots: &nrots,
                    eTableSelf: &eTableSelf,
                    eTablePair: eTablePair,
                    bestrot: &bestrot
                )
                threshold *= 2
            }

            var graphs = [AdjGraph]()
            var visited = [Int](repeating: 0, count: nunfix)
            constructSubgraphs(
                nunfix: nunfix,
                unfixres: &unfixres,
                visited: &visited,
                adjMatrix: adjMatrix,
                flagMatrix: &flagMatrix,
                nrots: &nrots,
                eTableSelf: eTableSelf,
                bestrot: &bestrot,
                graphs: &graphs
            )

            // Remove newly fixed residues from unfixres
            unfixres.removeAll { nrots[$0] == 1 }

            for i1 in 0..<graphs.count {
                var graph = graphs[i1]
                var connBags = [TreeBag]()

                subgraph2TreeDecomposition(graph: &graph, connBags: &connBags)

                let treewidth = checkTreewidth(connBags: connBags)

                if treewidth <= TREEWIDTH_CUT {
                    treeDecompositionBottomToTopCalcEnergy(
                        connBags: &connBags,
                        nrots: nrots,
                        eTableSelf: eTableSelf,
                        eTablePair: eTablePair
                    )
                    rootBagFindGMEC(
                        rootbag: &connBags[0],
                        connBags: connBags,
                        nrots: &nrots,
                        eTableSelf: eTableSelf,
                        eTablePair: eTablePair,
                        bestrot: &bestrot
                    )
                    for childIdx in connBags[0].childIndices {
                        treeDecompositionTopToBottomAssignRotamer(
                            parbag: connBags[0],
                            childIdx: childIdx,
                            connBags: &connBags,
                            nrots: &nrots,
                            bestrot: &bestrot
                        )
                    }
                    // Remove solved sites from unfixres
                    for site in graph.keys {
                        unfixres.removeAll { $0 == site }
                    }
                } else {
                    hardmode = true
                }
            }

            if !hardmode { break }
        }

        return bestrot
    }

    // MARK: - DEE Search Driver

    /// Runs 3 rounds of (Goldstein until convergence, fix, fold-in, Split until convergence, fix, fold-in).
    /// Returns true if there are still unfixed residues.
    private static func deeSearch(
        pos: inout [Int],
        nres: Int,
        nrots: inout [Int],
        eTableSelf: inout [[Float]],
        eTablePair: inout [Int: [Int: [[Float]]]],
        bestrot: inout [Int]
    ) -> Bool {

        // We run 3 cycles of (Goldstein, Split) = 6 total stages
        for cycle in 0..<3 {
            // --- Goldstein phase ---
            if cycle > 0 {
                pos.removeAll()
                for i in 0..<nres {
                    if nrots[i] > 1 { pos.append(i) }
                }
            }

            var ndeadDEE = 1
            while ndeadDEE != 0 {
                ndeadDEE = deeGoldstein(pos: pos, nrots: nrots, eTableSelf: &eTableSelf, eTablePair: eTablePair)
            }

            var fixres = [Int]()
            var unfixres = [Int]()
            for i in 0..<pos.count {
                let ip = pos[i]
                var n = 0
                var rot = 0
                for j in 0..<nrots[ip] {
                    if eTableSelf[ip][j] < DEAD_ENERGY {
                        rot = j
                        n += 1
                    }
                }
                if n == 1 {
                    nrots[ip] = 1
                    bestrot[ip] = rot
                    fixres.append(ip)
                } else if n > 1 {
                    unfixres.append(ip)
                }
            }
            if unfixres.isEmpty { return false }

            // Fold fixed-residue pair energies into self energies
            foldFixedIntoSelf(
                unfixres: unfixres,
                fixres: fixres,
                nrots: nrots,
                eTableSelf: &eTableSelf,
                eTablePair: eTablePair,
                bestrot: bestrot
            )

            // --- Split phase ---
            pos.removeAll()
            for i in 0..<nres {
                if nrots[i] > 1 { pos.append(i) }
            }

            ndeadDEE = 1
            while ndeadDEE != 0 {
                ndeadDEE = deeSplit(pos: pos, nrots: nrots, eTableSelf: &eTableSelf, eTablePair: eTablePair)
            }

            fixres.removeAll()
            unfixres.removeAll()
            for i in 0..<pos.count {
                let ip = pos[i]
                var n = 0
                var rot = 0
                for j in 0..<nrots[ip] {
                    if eTableSelf[ip][j] < DEAD_ENERGY {
                        rot = j
                        n += 1
                    }
                }
                if n == 1 {
                    nrots[ip] = 1
                    bestrot[ip] = rot
                    fixres.append(ip)
                } else if n > 1 {
                    unfixres.append(ip)
                }
            }
            if unfixres.isEmpty { return false }

            foldFixedIntoSelf(
                unfixres: unfixres,
                fixres: fixres,
                nrots: nrots,
                eTableSelf: &eTableSelf,
                eTablePair: eTablePair,
                bestrot: bestrot
            )
        }

        return true
    }

    /// Fold pair energies from fixed residues into unfixed residues' self energies.
    private static func foldFixedIntoSelf(
        unfixres: [Int],
        fixres: [Int],
        nrots: [Int],
        eTableSelf: inout [[Float]],
        eTablePair: [Int: [Int: [[Float]]]],
        bestrot: [Int]
    ) {
        for ipos in unfixres {
            for jpos in fixres {
                guard let pairIJ = eTablePair[ipos]?[jpos] else { continue }
                let rot = bestrot[jpos]
                for k in 0..<nrots[ipos] {
                    if eTableSelf[ipos][k] < DEAD_ENERGY {
                        eTableSelf[ipos][k] += pairIJ[k][rot]
                    }
                }
            }
        }
    }

    // MARK: - DEE Goldstein

    private static func deeGoldstein(
        pos: [Int],
        nrots: [Int],
        eTableSelf: inout [[Float]],
        eTablePair: [Int: [Int: [[Float]]]]
    ) -> Int {
        var elimination = 0
        for i in 0..<pos.count {
            let ip = pos[i]
            for s in 0..<nrots[ip] {
                if eTableSelf[ip][s] > DEAD_ENERGY { continue }
                for r in 0..<nrots[ip] {
                    if r == s { continue }
                    if eTableSelf[ip][r] > DEAD_ENERGY { continue }
                    var ex = eTableSelf[ip][s] - eTableSelf[ip][r]
                    for j in 0..<pos.count {
                        if j == i { continue }
                        let jp = pos[j]
                        guard let pairIJ = eTablePair[ip]?[jp] else { continue }
                        var ey: Float = 1e8
                        for t in 0..<nrots[jp] {
                            if eTableSelf[jp][t] > DEAD_ENERGY { continue }
                            let em = pairIJ[s][t] - pairIJ[r][t]
                            if em < ey { ey = em }
                        }
                        ex += ey
                    }
                    if ex > DEE_THRESHOLD {
                        eTableSelf[ip][s] = DEAD_MARK
                        elimination += 1
                        break  // rotamer s eliminated, move to next s
                    }
                }
            }
        }
        return elimination
    }

    // MARK: - DEE Split

    private static func deeSplit(
        pos: [Int],
        nrots: [Int],
        eTableSelf: inout [[Float]],
        eTablePair: [Int: [Int: [[Float]]]]
    ) -> Int {
        var elimination = 0
        for i in 0..<pos.count {
            let ip = pos[i]
            for s in 0..<nrots[ip] {
                if eTableSelf[ip][s] > DEAD_ENERGY { continue }

                // Precompute min_t(pair[s,t] - pair[r,t]) for each (r, j)
                var storeYjr = [[Float]](repeating: [Float](repeating: 0, count: pos.count), count: nrots[ip])
                for r in 0..<nrots[ip] {
                    if r == s { continue }
                    if eTableSelf[ip][r] > DEAD_ENERGY { continue }
                    for j in 0..<pos.count {
                        if j == i { continue }
                        let jp = pos[j]
                        guard let pairIJ = eTablePair[ip]?[jp] else { continue }
                        var ey: Float = 1e8
                        for t in 0..<nrots[jp] {
                            if eTableSelf[jp][t] > DEAD_ENERGY { continue }
                            let em = pairIJ[s][t] - pairIJ[r][t]
                            if em < ey { ey = em }
                        }
                        storeYjr[r][j] = ey
                    }
                }

                var eliminated = false
                for k in 0..<pos.count {
                    if k == i { continue }
                    let kp = pos[k]
                    guard let pairIK = eTablePair[ip]?[kp] else { continue }

                    var elim = [Int](repeating: 0, count: nrots[kp])
                    for r in 0..<nrots[ip] {
                        if r == s { continue }
                        if eTableSelf[ip][r] > DEAD_ENERGY { continue }

                        var ex = eTableSelf[ip][s] - eTableSelf[ip][r]
                        for j in 0..<pos.count {
                            if j == i || j == k { continue }
                            guard eTablePair[ip]?[pos[j]] != nil else { continue }
                            ex += storeYjr[r][j]
                        }
                        for v in 0..<nrots[kp] {
                            if eTableSelf[kp][v] > DEAD_ENERGY { continue }
                            if ex + pairIK[s][v] - pairIK[r][v] > DEE_THRESHOLD {
                                elim[v] = 1
                            }
                        }
                    }
                    let allElim = elim.allSatisfy { $0 != 0 }
                    if allElim {
                        eTableSelf[ip][s] = DEAD_MARK
                        elimination += 1
                        eliminated = true
                        break  // goto FLAG_SPLIT equivalent
                    }
                }
                if eliminated { continue }
            }
        }
        return elimination
    }

    // MARK: - Graph Construction

    private static func constructAdjMatrix(
        nunfix: Int,
        unfixres: [Int],
        nrots: inout [Int],
        eTableSelf: [[Float]],
        eTablePair: inout [Int: [Int: [[Float]]]],
        bestrot: inout [Int],
        adjMatrix: inout [[Int]]
    ) {
        // Build adjacency: edge if any pair energy exceeds EPAIR_CUT
        for i in 0..<nunfix - 1 {
            let ipos = unfixres[i]
            for j in (i + 1)..<nunfix {
                let jpos = unfixres[j]
                guard let pairIJ = eTablePair[ipos]?[jpos] else { continue }
                var found = false
                for k in 0..<nrots[ipos] {
                    if eTableSelf[ipos][k] > DEAD_ENERGY { continue }
                    for l in 0..<nrots[jpos] {
                        if eTableSelf[jpos][l] > DEAD_ENERGY { continue }
                        if pairIJ[k][l] > EPAIR_CUT || pairIJ[k][l] < -EPAIR_CUT {
                            adjMatrix[i][j] = 1
                            adjMatrix[j][i] = 1
                            found = true
                            break
                        }
                    }
                    if found { break }
                }
                if !found {
                    // Remove weak edge from pair table
                    eTablePair[ipos]?[jpos] = nil
                }
            }
        }

        // Remove isolated residues (no edges): pick lowest self-energy rotamer
        for i in 0..<nunfix {
            let allZeros = adjMatrix[i].allSatisfy { $0 == 0 }
            if allZeros {
                var emin: Float = 1e8
                var rot = 0
                for r in 0..<nrots[unfixres[i]] {
                    if eTableSelf[unfixres[i]][r] < emin {
                        emin = eTableSelf[unfixres[i]][r]
                        rot = r
                    }
                }
                nrots[unfixres[i]] = 1
                bestrot[unfixres[i]] = rot
            }
        }
    }

    private static func constructSubgraphs(
        nunfix: Int,
        unfixres: inout [Int],
        visited: inout [Int],
        adjMatrix: [[Int]],
        flagMatrix: inout [[Int]],
        nrots: inout [Int],
        eTableSelf: [[Float]],
        bestrot: inout [Int],
        graphs: inout [AdjGraph]
    ) {
        visited = [Int](repeating: 0, count: nunfix)
        for i in 0..<nunfix {
            if visited[i] == 0 {
                var newsg = AdjGraph()
                findSubgraphByDFS(
                    graph: &newsg,
                    u: i,
                    visited: &visited,
                    adjMatrix: adjMatrix,
                    flagMatrix: &flagMatrix,
                    unfixres: unfixres
                )
                if !newsg.isEmpty {
                    graphs.append(newsg)
                }

                // Single-node subgraph: fix by lowest self-energy
                if newsg.count == 1 {
                    let site = unfixres[i]
                    var rot = 0
                    var eMin: Float = 1000.0
                    for j in 0..<nrots[site] {
                        if eTableSelf[site][j] > DEAD_ENERGY { continue }
                        if eTableSelf[site][j] < eMin {
                            eMin = eTableSelf[site][j]
                            rot = j
                        }
                    }
                    bestrot[site] = rot
                    nrots[site] = 1
                }
            }
        }
    }

    private static func findSubgraphByDFS(
        graph: inout AdjGraph,
        u startU: Int,
        visited: inout [Int],
        adjMatrix: [[Int]],
        flagMatrix: inout [[Int]],
        unfixres: [Int]
    ) {
        var u = startU
        visited[u] = 1
        var stack = [u]
        while !stack.isEmpty {
            var hasEdge = false
            var w = -1
            for col in 0..<adjMatrix[u].count {
                if adjMatrix[u][col] == 1 && flagMatrix[u][col] == 0 {
                    hasEdge = true
                    w = col
                    break
                }
            }

            if !hasEdge {
                u = stack.removeLast()
            } else {
                visited[w] = 1
                stack.append(w)
                flagMatrix[u][w] = 1
                flagMatrix[w][u] = 1
                let uSite = unfixres[u]
                let wSite = unfixres[w]
                graph[uSite, default: []].insert(wSite)
                graph[wSite, default: []].insert(uSite)
            }
        }
    }

    // MARK: - Graph Edge Decomposition

    private static func graphEdgeDecomposition(
        adjMatrix: inout [[Int]],
        threshold: Float,
        unfixres: [Int],
        nrots: inout [Int],
        eTableSelf: inout [[Float]],
        eTablePair: [Int: [Int: [[Float]]]],
        bestrot: inout [Int]
    ) {
        let n = adjMatrix.count
        for i in 0..<n - 1 {
            let k = unfixres[i]
            for j in (i + 1)..<n {
                let l = unfixres[j]
                guard adjMatrix[i][j] == 1, let pairKL = eTablePair[k]?[l] else { continue }

                var countM = 0
                var countN = 0
                for m in 0..<nrots[k] {
                    if eTableSelf[k][m] <= DEAD_ENERGY { countM += 1 }
                }
                for nn in 0..<nrots[l] {
                    if eTableSelf[l][nn] <= DEAD_ENERGY { countN += 1 }
                }
                guard countM > 0 && countN > 0 else { continue }

                // Compute mean pair energy
                var abar: Float = 0
                for m in 0..<nrots[k] {
                    if eTableSelf[k][m] > DEAD_ENERGY { continue }
                    for nn in 0..<nrots[l] {
                        if eTableSelf[l][nn] > DEAD_ENERGY { continue }
                        abar += pairKL[m][nn]
                    }
                }
                abar /= (2.0 * Float(countM) * Float(countN))

                // Row means
                var ak = [Float](repeating: DEAD_MARK, count: nrots[k])
                for m in 0..<nrots[k] {
                    if eTableSelf[k][m] > DEAD_ENERGY { continue }
                    var temp: Float = 0
                    for nn in 0..<nrots[l] {
                        if eTableSelf[l][nn] > DEAD_ENERGY { continue }
                        temp += pairKL[m][nn]
                    }
                    temp /= Float(countN)
                    ak[m] = temp - abar
                }

                // Column means
                var bl = [Float](repeating: DEAD_MARK, count: nrots[l])
                for nn in 0..<nrots[l] {
                    if eTableSelf[l][nn] > DEAD_ENERGY { continue }
                    var temp: Float = 0
                    for m in 0..<nrots[k] {
                        if eTableSelf[k][m] > DEAD_ENERGY { continue }
                        temp += pairKL[m][nn]
                    }
                    temp /= Float(countM)
                    bl[nn] = temp - abar
                }

                // Max deviation
                var maxdev: Float = -1e8
                for m in 0..<nrots[k] {
                    if eTableSelf[k][m] > DEAD_ENERGY { continue }
                    for nn in 0..<nrots[l] {
                        if eTableSelf[l][nn] > DEAD_ENERGY { continue }
                        let dev = abs(pairKL[m][nn] - ak[m] - bl[nn])
                        if dev > maxdev { maxdev = dev }
                    }
                }

                if maxdev <= threshold {
                    adjMatrix[i][j] = 0
                    adjMatrix[j][i] = 0
                    for m in 0..<nrots[k] {
                        if eTableSelf[k][m] > DEAD_ENERGY { continue }
                        eTableSelf[k][m] += ak[m]
                    }
                    for nn in 0..<nrots[l] {
                        if eTableSelf[l][nn] > DEAD_ENERGY { continue }
                        eTableSelf[l][nn] += bl[nn]
                    }
                }
            }
        }

        // Remove residues with no remaining edges
        for i in 0..<n {
            let k = unfixres[i]
            var nonzero = false
            for j in 0..<n {
                if adjMatrix[i][j] != 0 { nonzero = true; break }
            }
            if !nonzero {
                var emin: Float = 1e8
                var best = 0
                for j in 0..<nrots[k] {
                    if eTableSelf[k][j] > DEAD_ENERGY { continue }
                    if eTableSelf[k][j] < emin {
                        emin = eTableSelf[k][j]
                        best = j
                    }
                }
                bestrot[k] = best
                nrots[k] = 1
            }
        }
    }

    // MARK: - Tree Decomposition

    /// Build tree decomposition from a subgraph.
    /// Populates connBags with the connected bag tree.
    private static func subgraph2TreeDecomposition(
        graph: inout AdjGraph,
        connBags: inout [TreeBag]
    ) {
        // 1. Sort vertices by degree (ascending)
        struct VertexEntry {
            var idx: Int
            var neighbors: Set<Int>
        }

        var sortgraph = [VertexEntry]()
        for (vertex, neibs) in graph {
            sortgraph.append(VertexEntry(idx: vertex, neighbors: neibs))
        }
        sortgraph.sort { $0.neighbors.count < $1.neighbors.count }

        // 2. Construct bags by greedy elimination
        var bags = [TreeBag]()
        while !sortgraph.isEmpty {
            var newbag = TreeBag()
            let st = sortgraph[0]
            newbag.right.insert(st.idx)
            newbag.total.insert(st.idx)
            for nb in st.neighbors {
                newbag.left.insert(nb)
                newbag.total.insert(nb)
            }
            bags.append(newbag)

            // Remove the vertex from the sorted graph
            let removedVertex = st.idx
            sortgraph.removeFirst()

            // Remove edges to the removed vertex from remaining entries
            for idx in (0..<sortgraph.count).reversed() {
                if newbag.left.contains(sortgraph[idx].idx) {
                    sortgraph[idx].neighbors.remove(removedVertex)
                    if sortgraph[idx].neighbors.isEmpty {
                        sortgraph.remove(at: idx)
                    }
                }
            }

            // Add edges between all pairs of vertices in left (fill-in)
            if newbag.left.count > 1 {
                let leftArray = Array(newbag.left)
                // Ensure all left vertices exist in sortgraph
                for v in leftArray {
                    if !sortgraph.contains(where: { $0.idx == v }) {
                        sortgraph.append(VertexEntry(idx: v, neighbors: []))
                    }
                }
                // Add edges between all pairs
                for a in 0..<leftArray.count {
                    for b in (a + 1)..<leftArray.count {
                        let va = leftArray[a]
                        let vb = leftArray[b]
                        for idx in 0..<sortgraph.count {
                            if sortgraph[idx].idx == va {
                                sortgraph[idx].neighbors.insert(vb)
                                break
                            }
                        }
                        for idx in 0..<sortgraph.count {
                            if sortgraph[idx].idx == vb {
                                sortgraph[idx].neighbors.insert(va)
                                break
                            }
                        }
                    }
                }
            }

            // Re-sort by degree
            sortgraph.sort { $0.neighbors.count < $1.neighbors.count }
        }

        // 3. Connect bags into a tree (process from last bag to first = root is last)
        var vtsOnTree = Set<Int>()
        connBags.removeAll()
        var counter = 0

        while !bags.isEmpty {
            var bg = bags.removeLast()
            if counter == 0 {
                // Root bag
                bg.parentIdx = -1
                bg.type = .root
                connBags.append(bg)
                vtsOnTree.formUnion(bg.total)
            } else {
                let intersection = vtsOnTree.intersection(bg.total)
                for i in 0..<connBags.count {
                    let intersection2 = connBags[i].total.intersection(intersection)
                    if intersection2 == intersection {
                        bg.parentIdx = i
                        if connBags[i].type != .root {
                            connBags[i].type = .inner
                        }
                        connBags[i].childIndices.insert(connBags.count)
                        connBags.append(bg)
                        vtsOnTree.formUnion(bg.total)
                        break
                    }
                }
            }
            counter += 1
        }

        // Set child counters
        for i in 0..<connBags.count {
            connBags[i].childCounter = connBags[i].childIndices.count
        }
    }

    private static func checkTreewidth(connBags: [TreeBag]) -> Int {
        var width = 0
        for bag in connBags {
            if bag.total.count > width {
                width = bag.total.count
            }
        }
        // C++ does width++ at the end; this matches the original behavior
        width += 1
        return width
    }

    // MARK: - Deploy Sites Into Bag

    private static func bagDeploySites(
        bag: inout TreeBag,
        nrots: [Int],
        eTableSelf: [[Float]]
    ) {
        guard !bag.deployFlag else { return }
        bag.deployFlag = true

        bag.lsites = Array(bag.left).sorted()
        bag.tsites = bag.lsites
        bag.rsites = Array(bag.right).sorted()
        bag.tsites.append(contentsOf: bag.rsites)

        // Collect alive rotamers for left sites
        for site in bag.lsites {
            var rots = [Int]()
            for j in 0..<nrots[site] {
                if eTableSelf[site][j] > DEAD_ENERGY { continue }
                rots.append(j)
            }
            bag.lrots.append(rots)
        }

        // Collect alive rotamers for right sites
        for site in bag.rsites {
            var rots = [Int]()
            for j in 0..<nrots[site] {
                if eTableSelf[site][j] > DEAD_ENERGY { continue }
                rots.append(j)
            }
            bag.rrots.append(rots)
        }

        // Compute combination counts
        var lcount = 1
        for rots in bag.lrots { lcount *= rots.count }

        var rcount = 1
        for rots in bag.rrots { rcount *= rots.count }

        // Allocate DP tables
        bag.Etcom = [[Float]](repeating: [Float](repeating: 0, count: rcount), count: lcount)
        bag.Rtcom = [[Int]](repeating: [Int](repeating: -1, count: rcount), count: lcount)
    }

    // MARK: - Rotamer Combination Enumeration

    /// Enumerate all rotamer combinations for left sites.
    private static func getLeftBagRotamerCombination(
        bag: TreeBag,
        depth: Int,
        current: inout [Int],
        result: inout [[Int]]
    ) {
        if depth < bag.lsites.count {
            for roti in bag.lrots[depth] {
                current.append(roti)
                getLeftBagRotamerCombination(bag: bag, depth: depth + 1, current: &current, result: &result)
                current.removeLast()
            }
        } else {
            result.append(current)
        }
    }

    /// Enumerate right-bag rotamer combinations with self + intra-right pair energies.
    private static func calcRightBagRotamerCombinationEnergy(
        bag: TreeBag,
        depth: Int,
        Etmp: inout Float,
        Rtmp: inout [Int],
        Ercom: inout [Float],
        Rrcom: inout [[Int]],
        eTableSelf: [[Float]],
        eTablePair: [Int: [Int: [[Float]]]]
    ) {
        if depth < bag.rsites.count {
            let site = bag.rsites[depth]
            for roti in bag.rrots[depth] {
                let Eold = Etmp
                var Enew = eTableSelf[site][roti]
                for k in 0..<depth {
                    guard let pairKS = eTablePair[bag.rsites[k]]?[site] else { continue }
                    Enew += pairKS[Rtmp[k]][roti]
                }
                Etmp += Enew
                Rtmp.append(roti)
                calcRightBagRotamerCombinationEnergy(
                    bag: bag, depth: depth + 1, Etmp: &Etmp, Rtmp: &Rtmp,
                    Ercom: &Ercom, Rrcom: &Rrcom,
                    eTableSelf: eTableSelf, eTablePair: eTablePair
                )
                Rtmp.removeLast()
                Etmp = Eold
            }
        } else {
            Rrcom.append(Rtmp)
            Ercom.append(Etmp)
        }
    }

    /// Enumerate left-bag rotamer combinations with self + intra-left pair energies.
    private static func calcLeftBagRotamerCombinationEnergy(
        bag: TreeBag,
        depth: Int,
        Etmp: inout Float,
        Rtmp: inout [Int],
        Elcom: inout [Float],
        Rlcom: inout [[Int]],
        eTableSelf: [[Float]],
        eTablePair: [Int: [Int: [[Float]]]]
    ) {
        if depth < bag.lsites.count {
            let site = bag.lsites[depth]
            for roti in bag.lrots[depth] {
                let Eold = Etmp
                var Enew = eTableSelf[site][roti]
                // C++ references Rlcom[Rlcom.size()-1][k] which is the
                // partially-built combination in progress. In the recursive
                // scheme we use Rtmp[k] instead.
                for k in 0..<depth {
                    guard let pairKS = eTablePair[bag.lsites[k]]?[site] else { continue }
                    Enew += pairKS[Rtmp[k]][roti]
                }
                Etmp += Enew
                Rtmp.append(roti)
                calcLeftBagRotamerCombinationEnergy(
                    bag: bag, depth: depth + 1, Etmp: &Etmp, Rtmp: &Rtmp,
                    Elcom: &Elcom, Rlcom: &Rlcom,
                    eTableSelf: eTableSelf, eTablePair: eTablePair
                )
                Rtmp.removeLast()
                Etmp = Eold
            }
        } else {
            Rlcom.append(Rtmp)
            Elcom.append(Etmp)
        }
    }

    // MARK: - Leaf Bag Energy Calculation

    private static func leafBagCalcEnergy(
        bag: inout TreeBag,
        Rclcom: inout [[Int]],
        eTableSelf: [[Float]],
        eTablePair: [Int: [Int: [[Float]]]]
    ) {
        var Ercom = [Float]()
        var Rrcom = [[Int]]()
        var Ertmp: Float = 0
        var Rrtmp = [Int]()
        calcRightBagRotamerCombinationEnergy(
            bag: bag, depth: 0, Etmp: &Ertmp, Rtmp: &Rrtmp,
            Ercom: &Ercom, Rrcom: &Rrcom,
            eTableSelf: eTableSelf, eTablePair: eTablePair
        )

        var Rltmp = [Int]()
        getLeftBagRotamerCombination(bag: bag, depth: 0, current: &Rltmp, result: &Rclcom)

        for idx2 in 0..<Rclcom.count {
            var emin: Float = 1e8
            var Rrmin = [Int]()
            for idx3 in 0..<Rrcom.count {
                var eval: Float = 0
                for idx4 in 0..<Rclcom[idx2].count {
                    for idx5 in 0..<Rrcom[idx3].count {
                        if let pairLR = eTablePair[bag.lsites[idx4]]?[bag.rsites[idx5]] {
                            eval += pairLR[Rclcom[idx2][idx4]][Rrcom[idx3][idx5]]
                        }
                    }
                }
                eval += Ercom[idx3]
                if !bag.Etcom.isEmpty {
                    eval += bag.Etcom[idx2][idx3]
                }
                if eval < emin {
                    Rrmin = Rrcom[idx3]
                    emin = eval
                }
            }
            bag.Etcom[idx2][0] = emin

            // Record best total rotamer combination (left + right)
            var Rttmp = [Int]()
            for idx4 in 0..<Rclcom[idx2].count {
                Rttmp.append(Rclcom[idx2][idx4])
            }
            for idx5 in 0..<Rrmin.count {
                Rttmp.append(Rrmin[idx5])
            }
            bag.Rtcom[idx2] = Rttmp
        }
    }

    // MARK: - Combine Child Into Parent

    private static func subsetCheck(
        leafLsites: [Int],
        parTsites: [Int]
    ) -> [Int] {
        var indices = [Int]()
        for i in 0..<leafLsites.count {
            var found = -1
            for j in 0..<parTsites.count {
                if leafLsites[i] == parTsites[j] {
                    found = j
                }
            }
            indices.append(found)
        }
        return indices
    }

    private static func combineChildIntoParentBag(
        childBag: inout TreeBag,
        parentBag: inout TreeBag,
        Rclcom: [[Int]],
        eTableSelf: [[Float]],
        eTablePair: [Int: [Int: [[Float]]]]
    ) {
        var Ercom = [Float]()
        var Rrcom = [[Int]]()
        var Rlcom = [[Int]]()
        var Ertmp: Float = 0
        var Rrtmp = [Int]()
        var Rltmp = [Int]()

        calcRightBagRotamerCombinationEnergy(
            bag: parentBag, depth: 0, Etmp: &Ertmp, Rtmp: &Rrtmp,
            Ercom: &Ercom, Rrcom: &Rrcom,
            eTableSelf: eTableSelf, eTablePair: eTablePair
        )
        getLeftBagRotamerCombination(bag: parentBag, depth: 0, current: &Rltmp, result: &Rlcom)

        childBag.indices = subsetCheck(leafLsites: childBag.lsites, parTsites: parentBag.tsites)

        for idx2 in 0..<Rlcom.count {
            for idx3 in 0..<Rrcom.count {
                var ppartrot = [Int]()
                var ppartsite = [Int]()
                for j in 0..<childBag.indices.count {
                    if childBag.indices[j] >= parentBag.lsites.count {
                        let rIdx = childBag.indices[j] - parentBag.lsites.count
                        ppartrot.append(Rrcom[idx3][rIdx])
                        ppartsite.append(parentBag.rsites[rIdx])
                    } else {
                        ppartrot.append(Rlcom[idx2][childBag.indices[j]])
                        ppartsite.append(parentBag.lsites[childBag.indices[j]])
                    }
                }

                for j in 0..<Rclcom.count {
                    if ppartsite == childBag.lsites && ppartrot == Rclcom[j] {
                        parentBag.Etcom[idx2][idx3] += childBag.Etcom[j][0]
                    }
                }
            }
        }
    }

    // MARK: - Bottom-to-Top DP

    private static func treeDecompositionBottomToTopCalcEnergy(
        connBags: inout [TreeBag],
        nrots: [Int],
        eTableSelf: [[Float]],
        eTablePair: [Int: [Int: [[Float]]]]
    ) {
        if connBags.count == 1 {
            bagDeploySites(bag: &connBags[0], nrots: nrots, eTableSelf: eTableSelf)
        } else {
            while true {
                var leafCount = 0
                // Snapshot indices to avoid mutation issues
                let count = connBags.count
                for idx1 in 0..<count {
                    if connBags[idx1].type == .leaf {
                        bagDeploySites(bag: &connBags[idx1], nrots: nrots, eTableSelf: eTableSelf)
                        var Rclcom = [[Int]]()
                        leafBagCalcEnergy(
                            bag: &connBags[idx1], Rclcom: &Rclcom,
                            eTableSelf: eTableSelf, eTablePair: eTablePair
                        )

                        let parentIdx = connBags[idx1].parentIdx
                        bagDeploySites(bag: &connBags[parentIdx], nrots: nrots, eTableSelf: eTableSelf)
                        var childCopy = connBags[idx1]
                        combineChildIntoParentBag(
                            childBag: &childCopy,
                            parentBag: &connBags[parentIdx],
                            Rclcom: Rclcom,
                            eTableSelf: eTableSelf,
                            eTablePair: eTablePair
                        )
                        connBags[idx1] = childCopy

                        connBags[idx1].type = .none
                        connBags[parentIdx].childCounter -= 1
                        if connBags[parentIdx].childCounter == 0 && connBags[parentIdx].type != .root {
                            connBags[parentIdx].type = .leaf
                        }
                        leafCount += 1
                    }
                }
                if leafCount == 0 { break }
            }
        }
    }

    // MARK: - Root Bag Find GMEC

    private static func rootBagFindGMEC(
        rootbag: inout TreeBag,
        connBags: [TreeBag],
        nrots: inout [Int],
        eTableSelf: [[Float]],
        eTablePair: [Int: [Int: [[Float]]]],
        bestrot: inout [Int]
    ) {
        var Ercom = [Float]()
        var Rrcom = [[Int]]()
        var Elcom = [Float]()
        var Rlcom = [[Int]]()
        var Ertmp: Float = 0
        var Eltmp: Float = 0
        var Rrtmp = [Int]()
        var Rltmp = [Int]()

        calcRightBagRotamerCombinationEnergy(
            bag: rootbag, depth: 0, Etmp: &Ertmp, Rtmp: &Rrtmp,
            Ercom: &Ercom, Rrcom: &Rrcom,
            eTableSelf: eTableSelf, eTablePair: eTablePair
        )
        calcLeftBagRotamerCombinationEnergy(
            bag: rootbag, depth: 0, Etmp: &Eltmp, Rtmp: &Rltmp,
            Elcom: &Elcom, Rlcom: &Rlcom,
            eTableSelf: eTableSelf, eTablePair: eTablePair
        )

        rootbag.EGMEC = 1e8
        for idx2 in 0..<Rlcom.count {
            for idx3 in 0..<Rrcom.count {
                var energy = Elcom[idx2]
                for idx4 in 0..<Rlcom[idx2].count {
                    for idx5 in 0..<Rrcom[idx3].count {
                        if let pairLR = eTablePair[rootbag.lsites[idx4]]?[rootbag.rsites[idx5]] {
                            energy += pairLR[Rlcom[idx2][idx4]][Rrcom[idx3][idx5]]
                        }
                    }
                }
                energy += Ercom[idx3]
                if !rootbag.Etcom.isEmpty {
                    energy += rootbag.Etcom[idx2][idx3]
                }

                if energy < rootbag.EGMEC {
                    rootbag.EGMEC = energy
                    rootbag.RLGMEC = Rlcom[idx2]
                    rootbag.RRGMEC = Rrcom[idx3]
                }
            }
        }

        // Set optimal rotamer indices
        for i in 0..<rootbag.lsites.count {
            nrots[rootbag.lsites[i]] = 1
            bestrot[rootbag.lsites[i]] = rootbag.RLGMEC[i]
        }
        for i in 0..<rootbag.rsites.count {
            nrots[rootbag.rsites[i]] = 1
            bestrot[rootbag.rsites[i]] = rootbag.RRGMEC[i]
        }
    }

    // MARK: - Top-to-Bottom Assignment

    private static func treeDecompositionTopToBottomAssignRotamer(
        parbag: TreeBag,
        childIdx: Int,
        connBags: inout [TreeBag],
        nrots: inout [Int],
        bestrot: inout [Int]
    ) {
        var childbag = connBags[childIdx]

        var ppartrot = [Int]()
        var ppartsite = [Int]()
        for j in 0..<childbag.indices.count {
            if childbag.indices[j] >= parbag.lsites.count {
                let rIdx = childbag.indices[j] - parbag.lsites.count
                ppartrot.append(parbag.RRGMEC[rIdx])
                ppartsite.append(parbag.rsites[rIdx])
            } else {
                ppartrot.append(parbag.RLGMEC[childbag.indices[j]])
                ppartsite.append(parbag.lsites[childbag.indices[j]])
            }
        }

        for j in 0..<childbag.Rtcom.count {
            let Rcltmp = Array(childbag.Rtcom[j].prefix(childbag.lsites.count))
            let Rcrtmp = Array(childbag.Rtcom[j].suffix(from: childbag.lsites.count))
            if ppartsite == childbag.lsites && ppartrot == Rcltmp {
                childbag.RLGMEC = Rcltmp
                childbag.RRGMEC = Rcrtmp
                for p in 0..<childbag.rsites.count {
                    nrots[childbag.rsites[p]] = 1
                    bestrot[childbag.rsites[p]] = childbag.RRGMEC[p]
                }
                break
            }
        }

        connBags[childIdx] = childbag

        for grandchildIdx in childbag.childIndices {
            treeDecompositionTopToBottomAssignRotamer(
                parbag: connBags[childIdx],
                childIdx: grandchildIdx,
                connBags: &connBags,
                nrots: &nrots,
                bestrot: &bestrot
            )
        }
    }
}
