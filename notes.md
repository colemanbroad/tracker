
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
- [ ] routines to rasterize continuous shapes
- [ ] show flow direction in tracking images
- [ ] Speed up delaunay
- [ ] delaunay 3D
- [ ] use DAG for temporal edge graph. enable tracking multiple timepoints.
- [ ] use grid_hash for faster NN lookup when building DAG.
- [ ] expand $c=c_0 + |dx1-dx0|^2 + |dx2-dx0|^2$ and simplify
- [ ] repeat greedy tracking for multiple (all?) initial vertices. combine with median + conflict resolution.

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