
"Geometric Data Structures for Computer Graphics" →  "Geometric Proximity Graph"

# Literature

[Info on numpy and ctypes](https://stackoverflow.com/questions/14887378/how-to-return-array-from-c-function-to-python-using-ctypes)
~~"Graph Matching"~~ →  "Robust Point Matching" or "Correspondence"
- Vector Field Consensus (ma_robust_2014)
- Point Set Registration: Coherent Point Drift (myronenko_point_2010)
- A new point matching algorithm for non-rigid registration (chui2003new)

# Todo

- [x] Fixed delaunay memory issue. Can use > 10k pts.
- [x] python interop and DLL reloading
- [x] memory management. preallocate numpy arrays.
- [x] greedy tracking that satisfies constraints
- [x] greedy strain cost that grows outwards from initial guess
- [x] make greedy tracking work in 2D/3D
- [x] priority queue for next vertex to choose
- [x] Evaluate actual tracking performance
- [x] replace O(n^2)-space container for temporal edges
- [x] fast spatial nn data struct. grid hash. 
- [ ] Speed up delaunay
- [ ] routines to rasterize continuous shapes
- [ ] show flow direction in tracking images
- [ ] delaunay 3D
- [ ] use DAG for temporal edge graph. enable tracking multiple timepoints.
- [ ] use grid_hash for faster NN lookup when building DAG.
- [ ] expand $c=c_0 + |dx1-dx0|^2 + |dx2-dx0|^2$ and simplify
- [ ] repeat greedy tracking for multiple (all?) initial vertices. combine with median + conflict resolution.
- [ ] "debug mode" for rasterizers/images which helps with subpixel precision (draws everything at 10x ? uses SVG ?).

- [ ] vector median filter to clean up _any_ tracking
- [ ] Viterbi Alg, but efficient. Don't build whole array, just small graph.
- [ ] compare grid_hash / knn / d-projection / locality sensitive hashing. NOTE: d-projection hash may work well for cells on surface of ellipsoid.
- [ ] StarryNite
- [ ] Fast Matching
- [ ] Marching cubes



# Questions

- [ ] how to use dynamic / static `.a` or `.dylib` lib from zig ? 


# topics

## Very Fast StrainCost Tracking

## Rasterizing continuous shapes

I could do everything my self in my own little rendered and try to get everything pixel perfect. Or I could try to use an SVG library. Or TinyVG.

## equality testing

to know if two slices of Tri's contain the same set of Tris we can
1. put each tri into hashmap-a and hashmap-b
2. compute hash of hashmap-a and hashmap-b, they should be the same if the sets are the same.

We want a fast, collision-free hash of the 3 u32's in a Tri which is invariant to order / vertex permutation.
Then we want _another_ hash of this hashmap, (but one that recognizes keys as distinct).

Do we want the Tri objects to always be in a canonical form ? (i.e. sorted low-to-high) to make comparison easy?
We could do this by sorting at Tri-creation-time.
Or would we rather enforce canonical form only when inserting into a hashmap?
Can we sub-class AutoHashMap and override the `put` and `get` methods ?

## logging and debugging

each module should have separate log destination ? and an extra that's unified?


## Fast Delaunay

- [ ] profiling
- [ ] multiple implementations
- [ ] speed test suite
- [ ] more efficient way to find first conflicting triangle?
- [ ] better geometric datastructure with fewer hashes.
    - [ ] maybe a `[pts][N_tri]u32` which maps each point to a list of triangles it is a part of. potentially faster / more space efficient than 
