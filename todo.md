
# Literature

[Info on numpy and ctypes](https://stackoverflow.com/questions/14887378/how-to-return-array-from-c-function-to-python-using-ctypes)
~~"Graph Matching"~~ →  "Robust Point Matching" or "Correspondence"
- Vector Field Consensus (ma_robust_2014)
- Point Set Registration: Coherent Point Drift (myronenko_point_2010)
- A new point matching algorithm for non-rigid registration (chui2003new)


# Tracking Overview

-[ ] visual comparison of overlap between two tracking proposals
-[ ] parse and track real ISBI data
-[ ] move cpnet3 tracking experiments to zig
-[ ] show flow direction in tracking images
-[ ] Implement the scoring functions in Zig (Det, Seg, Tra)
-[x] python interop and DLL reloading
-[x] memory management. preallocate numpy arrays.
-[skip] [Fast, brute force search of tracking solutions with brute force search.]
-[x] [Implement a Tracking type]
-[x] ~use DAG for temporal edge graph.~ enable tracking multiple timepoints 
-[skip] Implement a DetectionTimeseries type.
-[skip] Figure out how to move more complex types across the Zig / Python boundary.
-[x] Evaluate actual tracking performance

 # Tracking Methods

-[x] greedy tracking that satisfies constraints
-[x] greedy strain cost that grows outwards from initial guess
-[x] make greedy tracking work in 2D/3D
-[ ] [Hungarian Algorithm / Munkres]
-[ ] expand `c=c_0 + |dx1-dx0|^2 + |dx2-dx0|^2` and simplify

-[ ] repeat greedy tracking for multiple (all?) initial vertices. combine with median + conflict resolution.
-[ ] vector median filter to clean up _any_ tracking
-[ ] Viterbi tracker. Only consider local assignments = small transition matrix.
-[ ] StarryNite tracking
-[ ] Very Fast StrainCost Tracking
-[ ] compare grid_hash / knn / d-projection / locality sensitive hashing. NOTE: d-projection hash may work well for cells on surface of ellipsoid.

# Profiling 

-[x] Tracy profiling
-[x] XCode Instruments Profiling
-[x] Custom profiling
-[ ] speed test suite

# Nearest Neib

-[x] fast spatial nn data struct. grid hash / KDTree / other
    -[skip] [add index fields to KD Tree struct]
    -[ ] `findNearestNeibFromSortedList()` should first bin by Y then sort by X. This generalizes to higher dimensions.
-[ ] add k param to NN lookup datastructs
-[ ] replace voronoi with other NN lookup (e.g. knn)
-[ ] use grid_hash for faster NN lookup when building tracking DAG.

# Delaunay triangulation

-[x] Fixed delaunay memory issue. Can use > 10k pts.
-[x] priority queue for next vertex to choose
-[x] replace O(n^2)-space container for temporal edges
-[ ] more efficient way to find first conflicting triangle?
-[ ] delaunay 3D
-[ ] multiple implementations
-[ ] better geometric datastructure with fewer hashes.
    -[ ] maybe a `[pts][N_tri]u32` which maps each point to a list of triangles it is a part of. potentially faster
and more space efficient than ... 

# Algorithms

-[ ] Fast Matching
-[ ] Marching cubes

# Drawing 

-[ ] [Rasterizing continuous shapes]
-[ ] "debug mode" for rasterizers/images which helps with subpixel precision (draws everything at 10x ? uses SVG ?).

