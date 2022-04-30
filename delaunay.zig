const std = @import("std");
const geo = @import("geometry.zig");
// const grid_hash = @import("grid_hash2.zig");
const grid_hash = @import("tri_grid.zig");
const draw_mesh = @import("draw_mesh.zig");

fn waitForUserInput() !i64 {
    if (@import("builtin").is_test) return 0;

    const stdin = std.io.getStdIn().reader();
    const stdout = std.io.getStdOut().writer();

    var buf: [10]u8 = undefined;

    try stdout.print("Press 0 to quit: ", .{});

    if (try stdin.readUntilDelimiterOrEof(buf[0..], '\n')) |user_input| {
        const res = std.fmt.parseInt(i64, user_input, 10) catch return 1;
        if (res == 0) return 0;
    }
    return 1;
}

const Vec2 = geo.Vec2;
const CircleR2 = geo.CircleR2;
const max = std.math.max;
const min = std.math.min;

// const print = std.debug.print;

const assert = std.debug.assert;

const print = std.debug.print;

pub fn print2(
    comptime src_info: std.builtin.SourceLocation,
    comptime fmt: []const u8,
    args: anytype,
) void {
    if (true) return;
    const s1 = comptime std.fmt.comptimePrint("{s}:{d}:{d} ", .{ src_info.file, src_info.line, src_info.column });
    std.debug.print(s1[41..] ++ fmt, args);
}

// var gpa = std.heap.GeneralPurposeAllocator(.{}){};
// const alloc = gpa.allocator();

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

test {
    std.testing.refAllDecls(@This());
}

test "delaunay. resize a thing" {
    const alloc = std.testing.allocator;
    var pts = try alloc.alloc(f32, 100);
    defer alloc.free(pts);
    pts = alloc.resize(pts, pts.len + 1).?;
}

test "delaunay. resize many times" {
    const alloc = std.testing.allocator;
    var count: u32 = 100;
    while (count < 10000) : (count += 100) {
        var pts = try alloc.alloc([2]f32, count);
        defer alloc.free(pts);

        for (pts) |*v, j| {
            const i = @intToFloat(f32, j);
            v.* = .{ i, i * i };
        }

        try testresize(pts);
        print2(
            @src(),
            "after resize...\npts = {d}\n",
            .{pts[pts.len - 3 ..]},
        );
        print2(
            @src(),
            "len={d}",
            .{pts.len},
        );
    }
}

fn testresize(_pts: [][2]f32) !void {
    const alloc = std.testing.allocator;
    var pts = alloc.resize(_pts, _pts.len + 3).?;
    defer _ = alloc.shrink(pts, pts.len - 3);
    pts[pts.len - 3] = .{ 1, 0 };
    pts[pts.len - 2] = .{ 1, 1 };
    pts[pts.len - 1] = .{ 1, 2 };
    print2(
        @src(),
        "pts[-3..] = {d}\n",
        .{pts[pts.len - 3 ..]},
    );
    print2(
        @src(),
        "in testresize() : len={d}",
        .{pts.len},
    );
}

fn lessThanDist(p0: Vec2, p_l: Vec2, p_r: Vec2) bool {
    // if (1 > 0) return true else return false;
    const dx_l = p0 - p_l;
    const d_l = @sqrt(@reduce(.Add, dx_l * dx_l));
    const dx_r = p0 - p_r;
    const d_r = @sqrt(@reduce(.Add, dx_r * dx_r));
    if (d_l < d_r) return true else return false;
}

// collapse repeated elements [1,1,1,2,2,3,3,1,1,4,4] → [1,2,3,1,4]
pub fn collapseAllRepeated(comptime T: type, mem: []T) []T {
    var write: u16 = 0;
    var read: u16 = 1;
    while (read < mem.len) : (read += 1) {
        if (mem[write] == mem[read]) continue;
        write += 1;
        mem[write] = mem[read];
    }
    return mem[0 .. write + 1];
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

    // SET UP INITIAL POINTS //

    var pts = try allo.alloc(Vec2, _pts.len + 3);
    defer allo.free(pts);
    for (_pts) |p, i| pts[i] = p;
    // const oldlen = @intCast(u32, _pts.len);
    // pts = try allo.realloc(pts, oldlen + 3);

    // pick points that form a triangle around the bounding box but don't go too wide
    const bbox = geo.boundsBBox(_pts);
    const box_width = bbox.x.hi - bbox.x.lo;
    const box_height = bbox.y.hi - bbox.y.lo;
    // const end = pts.len;
    const n_pt = @intCast(u32, pts.len);
    pts[n_pt - 3] = .{ bbox.x.lo - box_width * 0.1, bbox.y.lo - box_height * 0.1 };
    pts[n_pt - 2] = .{ bbox.x.lo - box_width * 0.1, bbox.y.hi + 2 * box_height };
    pts[n_pt - 1] = .{ bbox.x.hi + 2 * box_width, bbox.y.lo - box_height * 0.1 };

    // Sort by distance from bounding box corner to reduce bad_triangles.
    // UPDATE: not any faster.
    // std.sort.sort(Vec2, _pts, Vec2{ bbox.x.lo, bbox.y.lo }, lessThanDist);

    // PTS IS NOW FIXED. THE MEMORY DOESN'T CHANGE,


    // ALLOC MEMORY FOR ALGORITHM GLOBAL STATE

    // GridHash of triangles' conflicting areas
    var trigrid = try grid_hash.GridHash2.init(allo, pts, 2, 2, @intCast(u16,pts.len)*2);
    defer trigrid.deinit(allo);
    // easy access to triangle neibs via edges
    var mesh2d = try geo.Mesh2D.init(allo, pts);
    defer mesh2d.deinit();

    // ALLOC MEM FOR TEMPORARY STATE WHICH IS RE-INITIALIZED EACH LOOP.

    // queue to for triangle search
    const MyQueue = struct { q: [100]?Tri, head: u8, tail: u8 }; // TODO: `q`: ?Tri → Tri
    var search_queue = MyQueue{ .q = .{null} ** 100, .head = 0, .tail = 1 };
    // Label triangles that we visit in queue
    const Valid = enum { Good, Evil };
    var triangle_label = std.AutoArrayHashMap(Tri, Valid).init(allo);
    defer triangle_label.deinit();
    // slice of bad triangles
    var bad_triangles = std.ArrayList(Tri).init(allo);
    defer bad_triangles.deinit();
    // Holds edges on polygon border
    var polyedges = try std.ArrayList(Edge).initCapacity(allo, pts.len);
    defer polyedges.deinit();
    // Count edge occurrences each round to identify bounding polygon.
    var edge_label = std.AutoHashMap(Edge, u2).init(allo);
    defer edge_label.deinit();

    print2(
        @src(),
        "mesh2d items len = {d}\n",
        .{mesh2d.vs.items.len},
    );

    // INITIALIZE GLOBAL STATE //

    // Initialize with one big bounding triangle and add to grid and mesh
    const big_tri = Tri{ n_pt - 3, n_pt - 2, n_pt - 1 };

    // try triangles.put(big_tri, {});
    try trigrid.addTri(big_tri, pts);
    try mesh2d.addTris(&[_]Tri{big_tri});

    // mesh2d.show();

    // MAIN LOOP OVER (nonboundary) POINTS

    for (pts[0 .. pts.len - 3]) |p, idx_pt| {

        // RESET LOOP STATE
        print2(
            @src(),
            "\n\nLOOP BEGIN {}\n\n",
            .{idx_pt},
        );
        print2(
            @src(),
            "All Tris = {d}\n",
            .{try mesh2d.validTris(allo)},
        );

        search_queue.head = 0;
        search_queue.tail = 1;
        search_queue.q = undefined;
        triangle_label.clearRetainingCapacity();
        bad_triangles.clearRetainingCapacity();
        polyedges.clearRetainingCapacity();
        edge_label.clearRetainingCapacity();

        // TODO speed up by only looping over nearby triangles.
        // How can we _prove_ that a set of triangles are invalid?
        // We just need to check triangles starting from center until we have a unique polygon
        // of bad edges where the inner triangles each fail circumcircle test but the outer triangles don't.
        // A graph of triangles neighbours would allow us to do this quickly.
        // Whenever we add a triangle to the set, we know it's neighbours and can remember them.
        //
        // search_queue walk
        // 1. Use GridHash to find first conflict from small set of potential conflict triangles. Add to Q.
        // 2. while Q not empty
        // 3.   pop tri from Q
        // 4.   if visited(tri) continue
        // 5.   if tri contains pt:
        //          add tri edges to edgelist
        //          add tri-neibs to Q
        //          add tri to badtri
        //          remove tri→tri-neibs from neib map
        // 6. for e in edgelist: remove if it appears twice. make tri if it appears once.

        // quickly find triangle that conflicts with pt `p`
        // if (idx_pt==10) @breakpoint();



        // print2(@src(), "firsttri {d} \n", .{firsttri});

        // while Q isn't empty, local search and update triangle_label.
        search_queue.q[0] = try trigrid.getFirstConflict(p,pts);
        while (search_queue.head < search_queue.tail) {
            // get next triangle in Q and advance head
            const current_tri = search_queue.q[search_queue.head].?;
            search_queue.head += 1;

            // if we've already seen next tri, then continue, else add to visited
            var maybe_entry = try triangle_label.getOrPut(current_tri);
            if (maybe_entry.found_existing) continue;
            maybe_entry.value_ptr.* = .Good; // default good

            // test if current_tri conflicts with pt. if not then continue.
            const tripts = [3]Vec2{ pts[current_tri[0]], pts[current_tri[1]], pts[current_tri[2]] };
            if (!geo.pointInTriangleCircumcircle2d(p, tripts)) continue;

            // since it conflicts add it to bad_triangles
            maybe_entry.value_ptr.* = .Evil;

            // See which neibs of current_tri exist and add them to Q if unvisited.
            const neibs = mesh2d.getTriNeibs(current_tri);
            for (neibs) |t| {
                if (t == null) continue;
                const tri = geo.Mesh2D.sortTri(t.?);
                if (triangle_label.contains(tri)) continue;

                // exists and hasn't been visited yet, so add to Q
                search_queue.q[search_queue.tail] = tri;
                search_queue.tail += 1;
            }
        }

        // fill bad_triangles
        {
            var it = triangle_label.iterator();
            while (it.next()) |kv| {
                if (kv.value_ptr.* == .Good) continue;
                try bad_triangles.append(kv.key_ptr.*);
            }
        }

        print2(
            @src(),
            "bad triangles : {d} \n",
            .{bad_triangles.items},
        );

        print2(@src()," trigrid.getWc(p) = \n {d} \n" ,.{trigrid.getWc(p)[0..7].*});
        // Remove bad triangles from all datastructures
        for (bad_triangles.items) |tri| trigrid.remTri(tri, pts);
        try mesh2d.removeTris(bad_triangles.items);
        print2(@src()," trigrid.getWc(p) = \n {d} \n" ,.{trigrid.getWc(p)[0..7].*});

        // trigrid

        // GET BOUNDING POLYGON

        // count the number of occurrences of each edge in bad triangles. unique edges occur once.
        for (bad_triangles.items) |_tri| {
            const tri = geo.Mesh2D.sortTri(_tri);
            for (tri) |_, i| {
                const edge = geo.Mesh2D.sortEdge(Edge{ tri[i], tri[(i + 1) % 3] });
                const gop_res = try edge_label.getOrPut(edge);
                if (gop_res.found_existing) {
                    gop_res.value_ptr.* += 1;
                } else {
                    gop_res.value_ptr.* = 1;
                }
            }
        }
        // edges that occur once are added to polyedges
        var it = edge_label.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* == 1) try polyedges.append(kv.key_ptr.*);
        }
        // organize edges to make ordered polygon traversing boundary.
        var polygon = try allo.alloc([2]u32, polyedges.items.len);
        defer allo.free(polygon);
        for (polygon) |*v, i| v.* = polyedges.items[i];
        var head: u16 = 0;
        var tail: u16 = 1;
        while (tail < polygon.len) {
            const target = polygon[head][1];
            const compare = polygon[tail];
            if (compare[0] == target) {
                polygon[tail] = polygon[head + 1];
                polygon[head + 1] = compare;
                head += 1;
                tail = head + 1;
                continue;
            }
            if (compare[1] == target) {
                polygon[tail] = polygon[head + 1];
                polygon[head + 1] = .{ compare[1], compare[0] };
                head += 1;
                tail = head + 1;
                continue;
            }
            tail += 1;
        }
        if (head != polyedges.items.len - 1) unreachable; // we should pass through the entire chain exactly
        var polygon2 = try allo.alloc(u32, polygon.len);
        defer allo.free(polygon2);
        for (polygon2) |*v, i| v.* = polygon[i][0];

        try draw_mesh.rasterizeHighlightStuff(mesh2d, "boundary.tga", &.{p}, polygon, &.{});


        // Add new triangles to mesh and grid d.s.
        try mesh2d.addPointInPolygon(@intCast(u32, idx_pt), polygon2);
        // mesh2d.show();

        for (polygon2) |_, i| {
            const tri = geo.Mesh2D.sortTri(Tri{ polygon2[i], polygon2[(i + 1) % polygon2.len], @intCast(u32, idx_pt) });

            // const tripts = [3]Vec2{pts[tri[0]] , pts[tri[1]] , pts[tri[2]]};
            try trigrid.addTri(tri, pts);

        }


        // try draw_mesh.rasterize(mesh2d, "drawtemp.tga");
        const img_name = try std.fmt.allocPrint(allo, "delaunay-vid/img{d:0>3}.tga", .{idx_pt});
        defer allo.free(img_name);
        try draw_mesh.rasterizeHighlightTri(mesh2d, img_name, bad_triangles.items);
        // _ = try waitForUserInput();

        assert(mesh2d.ts.count()==trigrid.triset.count());
        print2(@src(),"counts : mesh2d {} trigrid {}\n",.{mesh2d.ts.count() , trigrid.triset.count()});
    }

    // put final set of valid triangles into []Tri
    var idx_valid: u32 = 0;
    var validtriangles = try allo.alloc(Tri, mesh2d.ts.count());
    var it = mesh2d.ts.keyIterator();
    while (it.next()) |tri_ptr| {
        // @compileLog("Tri Type :" , @TypeOf(tri_ptr.*));
        // remove triangles containing starting points
        if (std.mem.max(u32, tri_ptr) >= n_pt - 3) continue;
        validtriangles[idx_valid] = tri_ptr.*;
        idx_valid += 1;
    }

    validtriangles = try allo.realloc(validtriangles, idx_valid);
    // @src(),print("There were {d} valid out of / {d} total triangles. validtriangles has len={d}.\n", .{idx_valid, triangles.items.len, validtriangles.len},);

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

// test "delaunay. basic delaunay" {
pub fn main() !void {
    const alloc = std.testing.allocator;

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

    const triangles = try delaunay2d(alloc, verts);
    defer alloc.free(triangles);

    print2(
        @src(),
        "\n\nfound {} triangles on {} vertices\n",
        .{ triangles.len, nparticles },
    );
}
