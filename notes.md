# Sun Apr  3 11:43:02 2022

Fixed delaunay memory issue.
Can pass arrays of points to Zig and return a tracking!

[Info on numpy and ctypes](https://stackoverflow.com/questions/14887378/how-to-return-array-from-c-function-to-python-using-ctypes)

"Geometric Data Structures for Computer Graphics" →  "Geometric Proximity Graph"

~~"Graph Matching"~~ →  "Robust Point Matching" or "Correspondence"
- Vector Field Consensus (ma_robust_2014)
- Point Set Registration: Coherent Point Drift (myronenko_point_2010)
- A new point matching algorithm for non-rigid registration (chui2003new)

# 

# Todo

- [ ] memory management. don't pass back memory you allocated in Zig !
- [x] make greedy tracking work in 2D/3D
- [ ] better data structure for delaunay edges : NOTE: I think this construction is O(n^2) ? Yes, but it's complicated. O(pts x triangles). **maybe we could speed up delaunay with a spatial grid** ?
- [ ] better data structure for temporal edges
- [ ] better data structure for NN queries?
- [ ] priority queue for next vertex to choose
- [ ] if strain cost is sum of quadratic terms $c=c_0 + |dx1-dx0|^2 + |dx2-dx0|^2$ we should be able to expand it and simplify evaluation? The sum is also a quadratic, or?
- [ ] repeat this greedy tracking for multiple initial vertices (every initial vertex?), then take median result for each vertex. Then perform conflict resolution / constraint resolution.
- [ ] 

