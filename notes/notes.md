
"Geometric Data Structures for Computer Graphics" →  "Geometric Proximity Graph"


# Literature

[Info on numpy and ctypes](https://stackoverflow.com/questions/14887378/how-to-return-array-from-c-function-to-python-using-ctypes)
~~"Graph Matching"~~ →  "Robust Point Matching" or "Correspondence"
- Vector Field Consensus (ma_robust_2014)
- Point Set Registration: Coherent Point Drift (myronenko_point_2010)
- A new point matching algorithm for non-rigid registration (chui2003new)

# Todo

-[x] Fixed delaunay memory issue. Can use > 10k pts.
-[x] python interop and DLL reloading
-[x] memory management. preallocate numpy arrays.
-[x] greedy tracking that satisfies constraints
-[x] greedy strain cost that grows outwards from initial guess
-[x] make greedy tracking work in 2D/3D
-[x] priority queue for next vertex to choose
-[x] Evaluate actual tracking performance
-[x] replace O(n^2)-space container for temporal edges
-[x] fast spatial nn data struct. grid hash / KDTree / other
    -[ ] add index fields to KD Tree struct
    -[ ] `findNearestNeibFromSortedList()` should first bin by Y then sort by X. 



-[ ] Implement the scoring functions in Zig.
-[ ] Implement a Tracking type and a DetectionTimeseries type.
-[ ] Figure out how to move more complex types across the Zig / Python boundary.
-[ ] Fast, brute force search of tracking solutions with brute force search.

-[x] Tracy profiling
-[x] XCode Instruments Profiling
-[x] Custom profiling
-[ ] more efficient way to find first conflicting triangle?
-[ ] speed test suite
-[ ] delaunay 3D
-[ ] multiple implementations
-[ ] better geometric datastructure with fewer hashes.
    -[ ] maybe a `[pts][N_tri]u32` which maps each point to a list of triangles it is a part of. potentially faster / more space efficient than 

-[ ] routines to rasterize continuous shapes
-[ ] "debug mode" for rasterizers/images which helps with subpixel precision (draws everything at 10x ? uses SVG ?).

-[ ] show flow direction in tracking images
-[ ] use DAG for temporal edge graph. enable tracking multiple timepoints.
-[ ] use grid_hash for faster NN lookup when building tracking DAG.
-[ ] expand $c=c_0 + |dx1-dx0|^2 + |dx2-dx0|^2$ and simplify
-[ ] repeat greedy tracking for multiple (all?) initial vertices. combine with median + conflict resolution.
-[ ] vector median filter to clean up _any_ tracking
-[ ] Viterbi tracker. Only consider local assignments = small transition matrix.
-[ ] compare grid_hash / knn / d-projection / locality sensitive hashing. NOTE: d-projection hash may work well for cells on surface of ellipsoid.
-[ ] StarryNite
-[ ] Fast Matching
-[ ] Marching cubes



# Questions

-[x] how to use dynamic / static `.a` or `.dylib` lib from zig ? 

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


# Fast spatial NN data structures

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

# Tracy Integration and Profiling

## Trace.zig

Easy to install and use, but not nanosecond precise. I got bogus numbers for all functions (25..75us) which was off by two orders of magnitude.

## Tracy

Download tracy to a source folder in software-thirdparty/ and git update it.
Then build the server (it was already built on previous mac, nice).
Then when building the local project you have to do `addTracy(b,exe)` to add options to your CompileStep, and when building you have to use `zig build -Dtracy=/Users/broaddus/Desktop/software-thirdparty/tracy/`.

I've stashed a commit in the tracy repo that shows how to build on M1 mac. I should really have made a branch...

NOTE: Fix the size issue with `export TRACY_DPI_SCALE=1.0` see [here](https://github.com/wolfpld/tracy/issues/513)

OK I'm just about fed up with macos... Tracy works at random. I've managed to make it segfault my application whenever I run a .Debug build. Then when I ran a .RelaseFast it would ignore 1/3 or 1/4 of my tracing annotations! But at least these timings looks reasonable... Only 120ns / point with 10k particles?

I've learned that running the tests in an infinite loop is a good trick to avoid short running program problems.

Tracy works as long as I don't profile multiple funcs at once...

N=100_000 - when alternating between kdtree and brute_force
- `findNearestNeibBruteForce()` median=124us sigma=6us – OK This makes sense! It had been optimized out previously.
- `findNearestNeibKDTree()` median=1.29us, sigma=995ns 
- `findNearestNeibFromSortedList()` median=416ns sigma=215ns.

N=100k – kdtree alone
- `findNearestNeibKDTree()` median=708ns, sigma=901ns

## Instruments

So then I tried the builting `sample ...` command line utility and it apparently has no access to source info unless we build in .Debug mode... which totally throws off the timings! 
Use `renice` to improve the quality of the `sample` util. [see](https://gist.github.com/loderunner/36724cc9ee8db66db305#improving-your-sample)
So now I'm downloading 8GB XCode 14 just to be able to use their instrumentation tools... Which probably won't even work? I can't even install XCode 14... And XCode 12.5 does work with macos 12.5. And the Command Line Dev Tools don't even have Instruments. And then according to [this](https://stackoverflow.com/questions/11445619/profiling-c-on-mac-os-x) I'm not even supposed to use Instruments in 2022... But then I can't get the other thing (xctrace) to work... But then Instruments DOES WORK!!! And it knows my symbols! Why? 

But unfortunately it smooshes everything together in a call tree when we really want to see the entire call stack including line numbers! This is the only way. Because fn A might call fn B in multiple locations. And fn A might be called from multiple different parents in different locations! And all this context might matter.

## Hyperfine

Maybe the best way to profile is just to do it from the command line....
`/usr/bin/time -l zig-binary`
OK I'm using hyperfine and this also works. Although it's cumbersome...
But now I need to remove the infinite loop

NOTE: The ERRORs I found below are cases where the distance is exactly equal, i think.

```
~/Desktop/work/isbi/zig-tracker> ./zig-out/bin/exe-kdtree2d
expected 8.28876867e-02, found 8.40167552e-02
index 0 incorrect. expected 8.28876867e-02, found 8.40167552e-02
ERROR #2 PTS MISSING
query  { 0.08558578044176102, 0.2876039743423462 }
kdtree { 0.0828876867890358, 0.2890166938304901 }
brute  { 0.0828876867890358, 0.2890166938304901 }
sorted { 0.08401675522327423, 0.29021427035331726 }
expected 2.17668324e-01, found 1.99670672e-01
index 0 incorrect. expected 2.17668324e-01, found 1.99670672e-01
ERROR #1 PTS MISSING
query  { 0.2101004272699356, 0.03756137564778328 }
kdtree { 0.2176683247089386, 0.04700809717178345 }
brute  { 0.199670672416687, 0.043704163283109665 }
sorted { 0.199670672416687, 0.043704163283109665 }
expected 2.17668324e-01, found 1.99670672e-01
index 0 incorrect. expected 2.17668324e-01, found 1.99670672e-01
ERROR #2 PTS MISSING
query  { 0.2101004272699356, 0.03756137564778328 }
kdtree { 0.2176683247089386, 0.04700809717178345 }
brute  { 0.199670672416687, 0.043704163283109665 }
sorted { 0.199670672416687, 0.043704163283109665 }
expected 5.84149360e-01, found 5.79440653e-01
index 0 incorrect. expected 5.84149360e-01, found 5.79440653e-01
ERROR #1 PTS MISSING
query  { 0.5918391346931458, 0.7836174964904785 }
kdtree { 0.5841493606567383, 0.7933627367019653 }
brute  { 0.5794406533241272, 0.7830010652542114 }
sorted { 0.5841493606567383, 0.7933627367019653 }
```

## Summary

There are at least two different things we were trying to do by profiling.

1. Figure out exactly how long specific functions take to run.
2. Figure out where our program spends it's time and why.

Tracy tells us about the former and Instruments tells us about the latter. Tracy _does_ have a sampling profiler, but unfortunately it doesn't work on macos.

Hyperfine is for benchmarking command line programs, and not really for ns-precise timings on code blocks.

Why is trace.zig so off base? Was I doing something wrong??
Maybe the reason is because it calls std.log! It really shouldn't be writing to std out... but instead should be writing to memory and then dumping when we C-C SIG ABORT.

OK, Actually getting timings statistics for individual functions is really easy...
We just have to keep track of the timestamps ourselves!

    N = 100_000 N_trials=5_000 (no brute force. notice kdtree time!)
    kdtree mean      712.200 stddev      815.438
    sorted mean      737.600 stddev      502.309

    N = 100_000 N_trials=5_000 (brute force must clear the cache and kill KDTree perf!)
    kdtree mean     1258.200 stddev     1147.694
     brute mean   126145.563 stddev     5487.230
    sorted mean      710.800 stddev      812.363

This way we can avoid Tracy, Trace.zig, Instruments, profiler, python, etc... It's just way faster!

The cool thing we learn is that including brute slows down KDTree but not Sorted, because Sorted references the same data as brute force and probably doesn't require moving stuff in/out of cache!

And I know these times are correct because they agree with the Tracy times!
Using the official `std.time.Timer` gives more precise timings. But still I
think there's a significant error on each timestamp. Maybe about 40ns ? I 
wonder if this error exists in Tracy's timings ?


Storing the points sorted by X removes grid spacing param, and is a dense
representation, obviates grid cell to point index mapping or the
inline-storage with gaps idea.

```zig
const a = struct {a: f32, b:u8};
```

## Improving the custom tracer

In a perfect world I'd be able to get Tracy to work on macos. But in an even perfecter world
I'd have custom profiling tools that I could use for summarizing and profiling functions with
nanosecond (instruction count?) precision and full control. 


# Removing python

Calling zig tracking functions from python allows me to integrate into existing CPNet code for tracking, 
as well as the ISBI Tracking GT experiments that I've already performed. However, it might be easier to 
move those tracking solutions over to zig than to maintain two separate worlds... I could use python for
loading TIFFs and export point set and tracking data for each GT dataset... or build my own in zig... 

This would require 
-[ ] extracting centerpoints from TIFFs
-[ ] extracting linking information from `man_track.txt`
-[ ] implement DET/TRA tracking comparisons in zig on the new zig `Tracking` type
-[ ] make `Tracking` type generic over 2D / 3D. 

Benefits
- Put all the linking methods into one codebase which can grow to include any linking approach we can imagine.
- Make a shared library where these methods are easy to call from any other codebase.
- Make all algorithms easy to visualize (and debug).
- Integrate linking with a UI for curating and editing tracks, including in 3D.
- Trains my low-level and algorithmic coding skills.

# Implementing min-cost greedy-assignment linking with division-costs

The general idea is that we re-evaluate some (all?) of the assignment costs after each assignment. In particular this allows us to make the second assignment to a parent more expensive by adding a (fixed?) extra cost making translation preferable to division unless the benefit outways the added cost. 

We could achieve this particular goal by simply iterating in a different way. By iterating over parents and assigning the closest child (within a threshold) we reduce false divisions. Then once each parent has an assignment we iterate over the remaining children and assign them to the nearest parent (also within a threshold). _This is probably a better heuristic than the standard nearest-parent approach!_ And it doesn't require anything really fancy.

It probably makes sense to pick the shortest edges first in an edgeQ.

```
Edge = struct{v1,v2,cost}
alledges : []Edge = compute-all-edgecosts(vs0,vs1)
for e in alledges: Q.add(e).sortby(Edge.cost)

ps = set(parents)

// This doesn't work because a parent may have no
loop {
    e = Q.pop-smallest()
    if e.parent.children = [] {
        e.parent.children = [e.child]
        ps.remove(e.parent)
    }
    if ps.empty(): break
}


For each parent find it's cheapest child.
Order by cost.
Assign parent to child if that child hasn't been assigned yet.
If it has, then find the next cheapest.

loop {
    if (not all-parents-assigned) {
        assign unassigned-parent to nearest child or EXIT
    } else {
        assign unassigned-child to nearest parent or ENTRY
    }
}

```



iterating over children and assigning each to the nearest parent


