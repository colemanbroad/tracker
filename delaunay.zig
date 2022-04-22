const std = @import("std");
const geo = @import("geometry.zig");

pub const Vec2 = geo.Vec2;
const max = std.math.max;
const min = std.math.min;

const print = std.debug.print;

// var gpa = std.heap.GeneralPurposeAllocator(.{}){};
// const alloc = gpa.allocator();
const alloc = std.testing.allocator;

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

test {std.testing.refAllDecls(@This()); }

test "delaunay. resize a thing" {
    var pts = try alloc.alloc(f32, 100);
    defer alloc.free(pts);
    pts = alloc.resize(pts, pts.len + 1).?;
}

test "delaunay. resize many times" {
    var count: u32 = 100;
    while (count < 10000) : (count += 100) {
        var pts = try alloc.alloc([2]f32, count);
        defer alloc.free(pts);

        for (pts) |*v, j| {
            const i = @intToFloat(f32, j);
            v.* = .{ i, i * i };
        }

        try testresize(pts);
        print("after resize...\npts = {d}\n", .{pts[pts.len - 3 ..]});
        print("len={d}", .{pts.len});
    }
}

fn testresize(_pts: [][2]f32) !void {
    var pts = alloc.resize(_pts, _pts.len + 3).?;
    defer _ = alloc.shrink(pts, pts.len - 3);
    pts[pts.len - 3] = .{ 1, 0 };
    pts[pts.len - 2] = .{ 1, 1 };
    pts[pts.len - 1] = .{ 1, 2 };
    print("pts[-3..] = {d}\n", .{pts[pts.len - 3 ..]});
    print("in testresize() : len={d}", .{pts.len});
}

fn lessThanDist(p0: Vec2, p_l: Vec2, p_r: Vec2) bool {
    // if (1 > 0) return true else return false;
    const dx_l = p0 - p_l;
    const d_l = @sqrt(@reduce(.Add, dx_l * dx_l));
    const dx_r = p0 - p_r;
    const d_r = @sqrt(@reduce(.Add, dx_r * dx_r));
    if (d_l < d_r) return true else return false;
}


const Edge = [2]u32;
pub const Tri = [3]u32;

/// Goal: speed up triangle checking
/// Have: queue and technique iteration given first intersecting triangle
/// Need: fast access to _any_ intersecting triangle. (see TriangleHash.zig)
/// 

// for (points) |p| (nearby pts first)
//   tri0 = getOverlappingTri
//   add tri0 to tri_queue
//   while (tri_queue.next()) |tri_next|
//     mark all invalid
//   


/// Implementation of Bowyer-Watson algorithm for 2D tessellations
/// SORTS _PTS IN PLACE !
pub fn delaunay2d(allo: std.mem.Allocator, _pts: []Vec2) ![]Tri {

    const bbox = geo.boundsBBox(_pts);
    const box_width = bbox.x.hi - bbox.x.lo;
    const box_height = bbox.y.hi - bbox.y.lo;

    // Sort by distance from bounding box corner to reduce bad_triangles.
    // UPDATE: not any faster.
    // std.sort.sort(Vec2, _pts, Vec2{ bbox.x.lo, bbox.y.lo }, lessThanDist);

    var pts = try allo.alloc(Vec2, _pts.len);
    defer allo.free(pts);
    for (pts) |*p, i| p.* = _pts[i];

    // pick points that form a triangle around the bounding box but don't go too wide
    const oldlen = @intCast(u32, pts.len);
    pts = try allo.realloc(pts, oldlen + 3);
    pts[oldlen + 0] = .{ bbox.x.lo - box_width * 0.1, bbox.y.lo - box_height * 0.1 };
    pts[oldlen + 1] = .{ bbox.x.lo - box_width * 0.1, bbox.y.hi + 2 * box_height };
    pts[oldlen + 2] = .{ bbox.x.hi + 2 * box_width, bbox.y.lo - box_height * 0.1 };


    // pts is now FIXED. The memory doesn't change,
    // const PtIdx = u32;

    // TODO: this is not efficient or robust. likely to break depending on point distribution.
    // Use this to get neib of inserted point
    // const nsides = @floatToInt(u16, @sqrt(@intToFloat(f32, pts.len))/4 );
    // var gh = try GridHash.init(allo,nsides,nsides,20,pts[0..],null);
    // defer gh.deinit();
    // THIS FAILS because we only want to search through points that have ALREADY BEEN ADDED

    // how are we going to find the triangle containing the newly added point?
    // round 1. easy. there is only 1 tri and we know it contains pt.
    // round 2. - brute force. fast. can easily do circle test, but then we're back where we started!
    //          - GridHash, but search until we find an ADDED point...
    //          - rebuild GridHash on pts every time...
    //          - brute force until we get to sqrt(pts)... then grid hash.


    // Then need to map from points to triangles
    // var pt2tri = try allo.alloc([10]?Tria , pts.len);
    // defer allo.free(pt2tri);

    // must store "maybe" triangles so we can easily remove triangles by setting them to null.
    // var triangles = try std.ArrayList(?Tria).initCapacity(allo, pts.len * 100);
    // defer triangles.deinit();

    // use ?Tria because boundary tri has 2 neibs, internal has 3.
    // bad triangles removed on every iteration.
    // var triangle_neighbours = std.AutoHashMap(Tria,[3]?Tria).init(allo);
    // defer triangle_neighbours.deinit();

    // keep only valid triangles
    var triangles = std.AutoHashMap(Tri,void).init(allo);
    defer triangles.deinit();

    // easy access to triangle neibs via edges
    // var edge_to_tri = std.AutoHashMap(Edge,[2]?Tria).init(allo);
    // defer edge_to_tri.deinit();
    
    // queue to for triangle search
    // var search_queue   = std.Queue(Tria).init();
    // NO ALLOC REQUIRED

    // keep track of visited triangles
    // var search_visited = std.AutoHashMap(Tria,Bool).init(allo);
    // defer search_visited.deinit();

    // Holds invalid triangles that fail the circle test. Indexes into `triangles`.
    var bad_triangles = try std.ArrayList(Tri).initCapacity(allo, pts.len);
    defer bad_triangles.deinit();

    // Holds edges on polygon border
    var polyedges = try std.ArrayList(Edge).initCapacity(allo, pts.len);
    defer polyedges.deinit();

    // If bad triangle edge is unique, then add it to big polygon.
    // First count the number of occurrences of each edge. Then add edges with count==1 to polygon.
    // max must be 2. count > 2 is bug.
    var edgehash = std.AutoHashMap(Edge, u2).init(allo);
    defer edgehash.deinit();


    // Initialize with one big bounding triangle
    // triangles.appendAssumeCapacity(Tria{ oldlen + 0, oldlen + 1, oldlen + 2 });
    const big_tri = Tri{oldlen + 0, oldlen + 1, oldlen + 2};
    try triangles.put(big_tri,{});


    // MAIN LOOP OVER (nonboundary) POINTS
    for (pts[0 .. pts.len - 3]) |p, idx_pt| {

        // if pt in triangle, then triangle is bad.
        bad_triangles.clearRetainingCapacity();

        // TODO speed up by only looping over nearby triangles?
        // How can we _prove_ that a set of triangles are invalid?
        // We just need to check triangles starting from center until we have a unique polygon
        // of bad edges where the inner triangle fails circumcircle test but the outer triangle doesn't.
        // A graph of triangles neighbours would allow us to do this quickly...
        // Whenever we add a triangle to the set, we know it's neighbours and can remember them...
        //
        // search_queue walk
        // 1. add the first triangle to the queue. can we find the unique triangle containing a query point?
        // 2. while q not empty
        // 3.   pop tri from q
        // 4.   if visited(tri) continue
        // 5.   if tri contains pt: 
        //          add tri edges to edgelist
        //          add tri-neibs to q
        //          add tri to badtri
        //          remove tri→tri-neibs from neib map
        // 6. for e in edgelist: remove if it appears twice. make tri if it appears once.

        var it0 = triangles.keyIterator();
        while (it0.next()) |tri| {
            // @compileLog(tri , @TypeOf(tri.*));
            // const vtri = if (tri) |v| v else continue;
            const tripts = [3]Vec2{ pts[tri.*[0]], pts[tri.*[1]], pts[tri.*[2]] };
            // const delta = tripts[0] - p;
            // if (dot2(delta,delta)>2500) continue; // TODO: Can we make this algorithm more efficient by assuming a certain point density?
            if (geo.pointInTriangleCircumcircle2d(p, tripts)) {
                try bad_triangles.append(tri.*);
            }
            // }
        }

        // clear the map, but don't release the memory.
        edgehash.clearRetainingCapacity();

        // count the number of occurrences of each edge in bad triangles. unique edges occur once.
        for (bad_triangles.items) |tri| {
            // const tri = triangles.items[tri_idx].?; // we know tri_idx only refers to valid,bad triangles
            const v0 = tri[0];
            const v1 = tri[1];
            const v2 = tri[2];
            // sort edges
            const e0 = Edge{ min(v0, v1), max(v0, v1) };
            const e1 = Edge{ min(v1, v2), max(v1, v2) };
            const e2 = Edge{ min(v2, v0), max(v2, v0) };
            if (edgehash.get(e0)) |c| {
                try edgehash.put(e0, c + 1);
            } else try edgehash.put(e0, 1);
            if (edgehash.get(e1)) |c| {
                try edgehash.put(e1, c + 1);
            } else try edgehash.put(e1, 1);
            if (edgehash.get(e2)) |c| {
                try edgehash.put(e2, c + 1);
            } else try edgehash.put(e2, 1);
        }

        // edges that occur once are added to polyedges
        polyedges.clearRetainingCapacity();
        for (bad_triangles.items) |tri| {
            // const tri = triangles.items[tri_idx].?;
            const v0 = tri[0];
            const v1 = tri[1];
            const v2 = tri[2];
            const e0 = Edge{ min(v0, v1), max(v0, v1) };
            const e1 = Edge{ min(v1, v2), max(v1, v2) };
            const e2 = Edge{ min(v2, v0), max(v2, v0) };
            const c0 = edgehash.get(e0);
            if (c0.? == 1) {
                try polyedges.append(e0);
            }
            const c1 = edgehash.get(e1);
            if (c1.? == 1) {
                try polyedges.append(e1);
            }
            const c2 = edgehash.get(e2);
            if (c2.? == 1) {
                try polyedges.append(e2);
            }
        }

        // Remove bad triangles
        for (bad_triangles.items) |tri| {
            _ = triangles.remove(tri); // returns true on removal
        }


        for (polyedges.items) |edge| {

            const tri = Tri{ edge[0], edge[1], @intCast(u32, idx_pt) };
            try triangles.put(tri,{});
            
            // const current_tris = edge_to_tri.get(edge);
            // if (current_tris==null) {
            //     edge_to_tri.put(.{tri,null});
            //     continue;
            // }
            // for (current_tris) |*t| {
            //     if (t.*==null) {
            //         t.* = tri;
            //         continue :outer;
            //     }
            // }
            // unreachable;

        }

        // if (@intToFloat(f32, triangles.items.len) > @intToFloat(f32,triangles.capacity) * 0.9) triangles.items = removeNullFromList(?Tria, triangles.items);

        // try showdelaunaystate(pts,triangles,idx_pt); // save to image
        // print("lengths of things : {d}\n", .{idx_pt});
        // print("triangles {d}\n", .{triangles.items.len});
        // print("bad_triangles {d}\n", .{bad_triangles.items.len});
        // print("polyedges {d}\n", .{polyedges.items.len});

        // if (try checkDelaunay(idx_pt,pts,triangles.items)) @breakpoint();
    }

    // clean up
    var idx_valid: u32 = 0;
    var validtriangles = try allo.alloc(Tri, triangles.count());

    var it = triangles.keyIterator();
    while (it.next()) |tri| {
        // remove triangles containing starting points
        if (tri.*[0] >= oldlen or tri.*[1] >= oldlen or tri.*[2] >= oldlen) continue;
        validtriangles[idx_valid] = tri.*;
        idx_valid += 1;
    }

    // _ = try allo.realloc(pts,pts.len-3); // FREE extra points
    // validtriangles = validtriangles[0..idx_valid];
    validtriangles = try allo.realloc(validtriangles, idx_valid);
    // print("There were {d} valid out of / {d} total triangles. validtriangles has len={d}.\n", .{idx_valid, triangles.items.len, validtriangles.len});

    return validtriangles;
}

pub fn removeNullFromList(comptime T: type, arr: []T) []T {
    var idx: usize = 0;
    for (arr) |v| {
        if (v != null) {
            arr[idx] = v;
            idx += 1;
        }
    }
    return arr[0..idx];
}

const process = std.process;

test "delaunay. basic delaunay" {
    // pub fn main() !void {

    const nparticles = if (@import("builtin").is_test) 100 else blk: {
        var arg_it = try process.argsWithAllocator(alloc);
        _ = arg_it.skip(); // skip exe name
        const npts_str = arg_it.next() orelse "100";
        break :blk try std.fmt.parseUnsigned(usize, npts_str, 10);
    };

    // 10k requires too much memory for triangles. The scaling is nonlinear.
    // After change I can do 10k (0.9s) 20k (2.7s) 30k (6s) 40k (10s) ...
    const verts = try alloc.alloc(Vec2, nparticles);
    for (verts) |*v| v.* = .{ random.float(f32), random.float(f32) };
    defer alloc.free(verts); // changes size when we call delaunay2d() ...

    const triangles = try delaunay2d(alloc,verts);
    defer alloc.free(triangles);

    print("\n\nfound {} triangles on {} vertices\n", .{ triangles.len, nparticles });
}
