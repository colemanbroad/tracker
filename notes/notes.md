
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
- [ ] fast spatial nn data struct. grid hash. 

- [x] tracy profiling
- [ ] more efficient way to find first conflicting triangle?
- [ ] speed test suite
- [ ] delaunay 3D
- [ ] multiple implementations
- [ ] better geometric datastructure with fewer hashes.
    - [ ] maybe a `[pts][N_tri]u32` which maps each point to a list of triangles it is a part of. potentially faster / more space efficient than 

- [ ] routines to rasterize continuous shapes
- [ ] "debug mode" for rasterizers/images which helps with subpixel precision (draws everything at 10x ? uses SVG ?).

- [ ] show flow direction in tracking images
- [ ] use DAG for temporal edge graph. enable tracking multiple timepoints.
- [ ] use grid_hash for faster NN lookup when building tracking DAG.
- [ ] expand $c=c_0 + |dx1-dx0|^2 + |dx2-dx0|^2$ and simplify
- [ ] repeat greedy tracking for multiple (all?) initial vertices. combine with median + conflict resolution.
- [ ] vector median filter to clean up _any_ tracking
- [ ] Viterbi tracker. Only consider local assignments = small transition matrix.
- [ ] compare grid_hash / knn / d-projection / locality sensitive hashing. NOTE: d-projection hash may work well for cells on surface of ellipsoid.
- [ ] StarryNite
- [ ] Fast Matching
- [ ] Marching cubes



# Questions

- [x] how to use dynamic / static `.a` or `.dylib` lib from zig ? 

Link static lib with `exe.addObjectFile("vendor/libcurl/lib/libcurl.a");`
Link dynamic lib with 
```
exe.addIncludeDir("vendor/libcurl/include");
exe.addLibPath("vendor/libcurl/lib");
exe.linkSystemLibraryName("curl");
```



# Very Fast StrainCost Tracking

# Rasterizing continuous shapes

I could do everything my self in my own little rendered and try to get everything pixel perfect. Or I could try to use an SVG library. Or TinyVG.

# equality testing

to know if two slices of Tri's contain the same set of Tris we can
1. put each tri into hashmap-a and hashmap-b
2. compute hash of hashmap-a and hashmap-b, they should be the same if the sets are the same.

We want a fast, collision-free hash of the 3 u32's in a Tri which is invariant to order / vertex permutation.
Then we want _another_ hash of this hashmap, (but one that recognizes keys as distinct).

Do we want the Tri objects to always be in a canonical form ? (i.e. sorted low-to-high) to make comparison easy?
We could do this by sorting at Tri-creation-time.
Or would we rather enforce canonical form only when inserting into a hashmap?
Can we sub-class AutoHashMap and override the `put` and `get` methods ?

# logging and debugging

each module should have separate log destination ? and an extra that's unified?


# fast spatial nn data struct. grid hash.

Sep 22 2022
- `tri_trid.zig`, `grid_hash2.zig`, `grid_hash.zig` cleanup
- `drawing.zig`, `draw_mesh.zig`, `drawing_basic.zig` cleanup
- `drawing` vs `rasterize` ?
- `cam3d` vs `render3d` ?


The API I WANT is ...

```go

const edges:[][2]u32 = undefined
const gr = topology.Graph().fromEdgeList(edges)

const neib_list = gr.neibs(p)
if (gr.maxDegree() < 4) {}
if (gr.isRegular()) {}


const kd = euclidean.dim2.KDTree()
const p1 = kd.nearest_neib(p0)
const p_list = kd.neibs(p0,10)
const dx = euclidean.dist(p0,p1)

const gh = euclidean.dim3.GridHash()

```


## Balanced Grid Hash

May 30 2023

The gridhash is a great idea that makes it easy to find nearest neibs. But we don't know the optimal grid density ahead of time, so we have to guess or learn from the data, or determine it on the fly. I think we'll know the total number of points and the image shape ahead of time, so that allows us to compute an average density, which allows us to compute a bin size given an expected number of objects per bin. The number of objects won't change over the lifetime of this datastructure, which is also very useful. Alternatively, we can just sort the points by a single coordinate, which still helps with nearest neib search! Double-alternatively we can sort the points by a single coordinate within a grid window, then sort the grid squares into rows by the opposite coordinate, then sort the rows into a full grid by the 1st coordinate again.... This is like a grid hash. 

Within a grid cell we brute force compute distance to every point. 

We could sort the points in a dense array `[N]Pt` by gridcell, then have a `GridCell -> []Pt` function that knows where the points are for a given grid cell. 

We could sort by X, then break X into Nx percentiles where we determine the location of grid spacing by the real data. Then within the n-th percentile we sort by Y instead of X, then do the same thing breaking Y up into dynamically
spaced grid. Then we have a lookup that maps an arbitrary point (x,y) to first an x index, then a specific y index for that x. This allows us to have roughly the same number of objects in every grid cell by using dynamic spacing! And it's still very fast to build. see diagram.

```python

# query point
qp = [10.0, 11.2]
gh = GridHash(pts)
gh.nearest(qp,k=3)
# nearest() first maps qp.x to an x-index. Then maps qp.y to a y-index.
gh.x_lines # [0.0, 1.2, 3.5, 7.0] These lines should include include the domain bounds.
gh.y_lines[2] # [0.0, 3.4, 8.9, 12.] These lines should also include top/bottom domain bounds.



```

## KDTrees with visualization

!!! IF you want SDL window to show up you MUST poll an event first! `SDL_PollEvent(&e)`. 

OK, I get it now.
For all spatial nearest neib datastructures we need to _prove_ that we've checked
all the available volume that might contain the nearest point. This is easier when the volumes are rectilinear / square (and easy to access in an array), but
we can do the same thing in a tree. Whenever we check a region we add it's volume
to the list of checked volumes and eventually we can prove we've checked everywhere the nearest point might be.

But with KDTrees there's another way of viewing it... 
**We start off willing to (depth first) search the entire tree!** 
But we can prune any branch where we can prove that it only contains regions that are unnecessary to check, i.e. we can skip any branch (x < p.x) if |p.x - q.x| > current_radius! That's all we have to check! Otherwise it's a simple depth first search with pruning!

!!! OK, after some exhaustive bug fixing I've implemented the Pointer Tree version of a KDTree and it's ONLY THREE TIMES FASTER THAN BRUTE FORCE iteration on 10k points. And 25x faster on 100k pts. We even are clever about reducing the number of iterations required to find the true point. Instead of 10k checks we only need avg of 35 checks! But still these 35 take about the same amount of time... Why? Is it just because of pointer indirection?

And the absolute timing was about 75us / NN query with N=10k pts. This is 75ms to query 1k pts. Or about 7ns / point. That's between 20/30 ops / point. That's pretty good? Right ballpark.

Also, using ReleaseFast actually slowed the brute force NN timing down! Why is that?

OK, I've profiled all the variants and the sorted list approach is actually
slightly faster than KDTree and avoids a few hyperparams.

N=1k pts (1000 trials)
name                          | mean (ns) | std dev
findNearestNeibBruteForce     | 24_504    | 6_278
findNearestNeibKDTree         | 19_070    | 6_888
findNearestNeibFromSortedList | 18_082    | 4_705

N=10k pts (1000 trials)
name                          | mean (ns) | std dev
findNearestNeibBruteForce     | 75_815    | 16_411
findNearestNeibKDTree         | 18_462    | 6_820
findNearestNeibFromSortedList | 17_630    | 4_160

N=100k pts (1000 trials)
name                          | mean (ns) | std dev
findNearestNeibBruteForce     | 514_601   | 32_465
findNearestNeibKDTree         | 17_875    | 3_546
findNearestNeibFromSortedList | 17_405    | 2_578

Variants of `findNearestNeibFromSortedList` approach will work even for more complex cost functions.










