/// Delaunay Triangulations in 2D Euclidean Space using
/// The Bowyer-Watson Algorithm.
const std = @import("std");
const g = @import("geometry.zig");
const im = @import("image_base.zig");
// const ztracy = @import("ztracy");

const Allocator = std.mem.Allocator;
const Pix = @Vector(2, u32);
const Vec3 = g.Vec3;
const Range = g.Range;
const BBox = g.BBox;

// const root = @import("root");
// const test_artifacts = @import("root").thisDir() ++ "test-artifacts/";
const test_home = "/Users/broaddus/work/isbi/zig-tracker/test-artifacts/track/";

// const build_options = @import("build_options");
// build_options.

const expect = std.testing.expect;
const clipi = g.clipi;
const floor = std.math.floor;

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

const max = std.math.max;
const min = std.math.min;
const assert = std.debug.assert;
const print = std.debug.print;

const Pt = @Vector(2, f32);
const CircleR2 = g.CircleR2;
// const CircleR2 = struct { pt: Pt, r2: f32 };
const PtIdx = u32;
const Edge = [2]PtIdx;
const Tri = [3]PtIdx;

pub fn print2(
    comptime src_info: std.builtin.SourceLocation,
    comptime fmt: []const u8,
    args: anytype,
) void {
    if (true) return;
    const s1 = comptime std.fmt.comptimePrint("{s}:{d}:{d} ", .{ src_info.file, src_info.line, src_info.column });
    std.debug.print(s1[41..] ++ fmt, args);
}

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

test {
    std.testing.refAllDecls(@This());
}

/// Goal: speed up triangle checking
/// Have: queue and technique iteration given first intersecting triangle
/// Need: fast access to _any_ intersecting triangle. (see TriangleHash.zig)
///
/// for (points) |p| (nearby pts first)
///   tri0 = getOverlappingTri
///   add tri0 to tri_queue
///   while (tri_queue.next()) |tri_next|
///     mark all invalid
///
/// Implementation of Bowyer-Watson algorithm for 2D tessellations
pub fn delaunay2d(allo: std.mem.Allocator, _pts: []Pt) !Mesh2D {
    // const full_tracy_zone = ztracy.ZoneNC(@src(), "full", 0x00_ff_00_00);
    // defer full_tracy_zone.End(); // ztracy

    // Copy _pts to pts. Add space for new bounding tri.
    var pts = try allo.alloc(Pt, _pts.len + 3);
    defer allo.free(pts);
    for (_pts, 0..) |p, i| pts[i] = p;

    // Add points that form a triangle around the bounding box.
    const bbox = g.boundsBBox(_pts);
    const box_width = bbox.x.hi - bbox.x.lo;
    const box_height = bbox.y.hi - bbox.y.lo;
    const n_pt = @as(u32, @intCast(pts.len));
    pts[n_pt - 3] = .{ bbox.x.lo - box_width * 0.1, bbox.y.lo - box_height * 0.1 };
    pts[n_pt - 2] = .{ bbox.x.lo - box_width * 0.1, bbox.y.hi + 2 * box_height };
    pts[n_pt - 1] = .{ bbox.x.hi + 2 * box_width, bbox.y.lo - box_height * 0.1 };

    // PTS IS NOW FIXED. THE MEMORY DOESN'T CHANGE,

    // GridHash of triangles' conflicting areas
    // var trigrid = try grid_hash.GridHash2.init(allo, pts, 50, 50, 40); //@intCast(u16,pts.len)*2);
    // defer trigrid.deinit();

    // easy access to triangle neibs via edges
    var mesh2d = try Mesh2D.init(allo, pts);
    // RETURN IT! Don't deinit().

    // Temporary state which is re-initialized each loop..

    // queue for triangle search
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
    var polygon_edges = try std.ArrayList(Edge).initCapacity(allo, pts.len);
    defer polygon_edges.deinit();

    // Count edge occurrences each round to identify bounding polygon.
    var edge_label = std.AutoHashMap(Edge, u2).init(allo);
    defer edge_label.deinit();

    // print2(
    //     @src(),
    //     "mesh2d items len = {d}\n",
    //     .{mesh2d.vs.items.len},
    // );

    // INITIALIZE GLOBAL STATE //

    // Initialize bounding triangle in mesh2d.
    const big_tri = Tri{ n_pt - 3, n_pt - 2, n_pt - 1 };
    try mesh2d.addTri(big_tri);
    // mesh2d.show();

    // MAIN LOOP OVER (nonboundary) POINTS

    for (pts[0 .. pts.len - 3], 0..) |p, idx_pt| {
        // const loop_tracy_zone = ztracy.ZoneNC(@src(), "loop", 0x00_ff_00_00);
        // defer loop_tracy_zone.End(); // ztracy

        // RESET LOOP STATE
        // print2(
        //     @src(),
        //     "\n\nLOOP BEGIN {}\n\n",
        //     .{idx_pt},
        // );
        // print2(
        //     @src(),
        //     "All Tris = {d}\n",
        //     .{try mesh2d.validTris(allo)},
        // );

        // const clearCapacity_tracy_zone = ztracy.ZoneNC(@src(), "clearCapacity", 0x00_ff_00_00);
        search_queue.head = 0;
        search_queue.tail = 1;
        search_queue.q = undefined;
        triangle_label.clearRetainingCapacity();
        bad_triangles.clearRetainingCapacity();
        polygon_edges.clearRetainingCapacity();
        edge_label.clearRetainingCapacity();
        // clearCapacity_tracy_zone.End(); // ztracy

        // TODO: speed up by only looping over nearby triangles.
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

        // while Q isn't empty, local search and update triangle_label.
        // search_queue.q[0] = try trigrid.getFirstConflict(p,pts);

        // search_queue.q[0] = blk: {
        //     var it = mesh2d.ts.iterator();
        //     while (it.next()) |kv| {
        //         const tri = kv.key_ptr.*;
        //         const tripts = [3]Vec2{ pts[tri[0]], pts[tri[1]], pts[tri[2]] };
        //         if (geo.pointInTriangleCircumcircle2d(p, tripts)) {
        //             break :blk tri;
        //         }
        //     }
        //     unreachable;
        // };

        // while (search_queue.head < search_queue.tail) {
        //     // get next triangle in Q and advance head
        //     const current_tri = search_queue.q[search_queue.head].?;
        //     search_queue.head += 1;

        //     // if we've already seen next tri, then continue, else add to visited
        //     var maybe_entry = try triangle_label.getOrPut(current_tri);
        //     if (maybe_entry.found_existing) continue;
        //     maybe_entry.value_ptr.* = .Good; // default good

        //     // test if current_tri conflicts with pt. if not then continue.
        //     const tripts = [3]Vec2{ pts[current_tri[0]], pts[current_tri[1]], pts[current_tri[2]] };
        //     if (!geo.pointInTriangleCircumcircle2d(p, tripts)) continue;

        //     // since it conflicts add it to bad_triangles
        //     maybe_entry.value_ptr.* = .Evil;

        //     // See which neibs of current_tri exist and add them to Q if unvisited.
        //     const neibs = mesh2d.getTriNeibs(current_tri);
        //     for (neibs) |t| {
        //         if (t == null) continue;
        //         const tri = Mesh2D.sortTri(t.?);
        //         if (triangle_label.contains(tri)) continue;

        //         // exists and hasn't been visited yet, so add to Q
        //         search_queue.q[search_queue.tail] = tri;
        //         search_queue.tail += 1;
        //     }
        // }

        // fill bad_triangles
        // {
        //     var it = triangle_label.iterator();
        //     while (it.next()) |kv| {
        //         if (kv.value_ptr.* == .Good) continue;
        //         try bad_triangles.append(kv.key_ptr.*);
        //     }
        // }

        {
            // const bad_tri_tracy_zone = ztracy.ZoneNC(@src(), "bad_tri", 0x00_ff_00_00);
            // defer bad_tri_tracy_zone.End(); // ztracy

            var it = mesh2d.ts.iterator();
            while (it.next()) |kv| {
                const tri = kv.key_ptr.*;
                const tripts = [3]Pt{ pts[tri[0]], pts[tri[1]], pts[tri[2]] };
                if (g.pointInTriangleCircumcircle2d(p, tripts)) {
                    try bad_triangles.append(tri);
                }
            }
        }

        // print2(
        //     @src(),
        //     "bad triangles : {d} \n",
        //     .{bad_triangles.items},
        // );

        // print2(@src()," trigrid.getWc(p) = \n {d} \n" ,.{trigrid.getWc(p)[0..7].*});
        // Remove bad triangles from all datastructures
        // for (bad_triangles.items) |tri| trigrid.remTri(tri, pts);
        for (bad_triangles.items) |tri| {
            mesh2d.removeTri(tri);
        }

        // trigrid

        // GET BOUNDING POLYGON

        // const bound_poly_tracy_zone = ztracy.ZoneNC(@src(), "bound_poly", 0x00_ff_00_00);

        // count the number of occurrences of each edge in bad triangles. unique edges occur once.
        for (bad_triangles.items) |_tri| {
            const tri = Mesh2D.sortTri(_tri);
            for (tri, 0..) |_, i| {
                const edge = Mesh2D.sortEdge(Edge{ tri[i], tri[(i + 1) % 3] });
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
            if (kv.value_ptr.* == 1) try polygon_edges.append(kv.key_ptr.*);
        }

        // organize edges to make ordered polygon traversing boundary.
        // for (polygon) |*v, i| v.* = polyedges.items[i];
        var edges = polygon_edges.items;
        var head: u16 = 0;
        var tail: u16 = 1;
        while (tail < edges.len) {
            const target = edges[head][1];
            const compare = edges[tail];
            if (compare[0] == target) {
                edges[tail] = edges[head + 1];
                edges[head + 1] = compare;
                head += 1;
                tail = head + 1;
                continue;
            }
            if (compare[1] == target) {
                edges[tail] = edges[head + 1];
                edges[head + 1] = .{ compare[1], compare[0] };
                head += 1;
                tail = head + 1;
                continue;
            }
            tail += 1;
        }
        if (head != edges.len - 1) unreachable; // we should pass through the entire chain exactly

        // bound_poly_tracy_zone.End(); // ztracy

        // try draw_mesh.rasterizeHighlightStuff(mesh2d, "boundary.tga", &.{p}, polygon, &.{});

        // Add new triangles to mesh and grid d.s.
        // const addPointInPoly_tracy_zone = ztracy.ZoneNC(@src(), "addPointInPoly", 0x00_ff_00_00);
        for (edges) |e| {
            try mesh2d.addTri(.{ e[0], e[1], @as(u32, @intCast(idx_pt)) });
        }
        // addPointInPoly_tracy_zone.End(); // ztracy

    }

    // remove boundary points from mesh2d

    // for (trigrid.triset.)
    var it = mesh2d.ts.iterator();
    while (it.next()) |kv| {
        // test triangles for boundary vertices
        const tri = kv.key_ptr.*;
        if (containsAny(u32, &tri, &.{ n_pt - 3, n_pt - 2, n_pt - 1 })) {
            mesh2d.removeTri(tri);
        }
    }

    // cheap way to remove the last 3 vertices (big triangle vertices)
    mesh2d.vs.items.len -= 3;

    return mesh2d;

    // // put final set of valid triangles into []Tri
    // var idx_valid: u32 = 0;
    // var validtriangles = try allo.alloc(Tri, mesh2d.ts.count());
    // var it = mesh2d.ts.keyIterator();
    // while (it.next()) |tri_ptr| {
    //     // @compileLog("Tri Type :" , @TypeOf(tri_ptr.*));
    //     // remove triangles containing starting points
    //     if (std.mem.max(u32, tri_ptr) >= n_pt - 3) continue;
    //     validtriangles[idx_valid] = tri_ptr.*;
    //     idx_valid += 1;
    // }

    // validtriangles = try allo.realloc(validtriangles, idx_valid);
    // // @src(),print("There were {d} valid out of / {d} total triangles. validtriangles has len={d}.\n", .{idx_valid, triangles.items.len, validtriangles.len},);

    // return validtriangles;
}

fn containsAny(comptime T: type, list: []const T, badlist: []const T) bool {
    for (list) |v| {
        for (badlist) |b| {
            if (v == b) return true;
        }
    }
    return false;
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
    // const pixels_loop_tracy_zone = ztracy.ZoneNC(@src(), "main", 0x00_ff_00_00);
    // defer pixels_loop_tracy_zone.End();

    const alloc = std.testing.allocator;

    const nparticles = if (@import("builtin").is_test) 100 else blk: {
        var arg_it = try process.argsWithAllocator(alloc);
        _ = arg_it.skip(); // skip exe name
        const npts_str = arg_it.next() orelse "4000";
        break :blk try std.fmt.parseUnsigned(usize, npts_str, 10);
    };

    // 10k requires too much memory for triangles. The scaling is nonlinear.
    // After change I can do 10k (0.9s) 20k (2.7s) 30k (6s) 40k (10s) ...
    const verts = try alloc.alloc(Pt, nparticles);
    for (verts) |*v| v.* = .{ random.float(f32), random.float(f32) };
    defer alloc.free(verts); // changes size when we call delaunay2d() ...

    // const triangles = try delaunay2d(alloc, verts);
    // defer alloc.free(triangles);
    var mesh = try delaunay2d(alloc, verts);
    defer mesh.deinit();

    try mesh.rasterize("delaunay_result.tga");

    print(
        // @src(),
        "\n\nfound {} triangles on {} vertices\n",
        .{ mesh.ts.count(), nparticles },
    );
}

// const test_home = "/Users/broaddus/work/zig-tracker/test-artifacts/tri_grid/";

// pub fn print2(
//     comptime src_info: std.builtin.SourceLocation,
//     comptime fmt: []const u8,
//     args: anytype,
// ) void {
//     const s1 = comptime std.fmt.comptimePrint("{s}:{d}:{d} ", .{ src_info.file, src_info.line, src_info.column });
//     std.debug.print(s1[41..] ++ fmt, args);
// }

// pub fn thisDir() []const u8 {
//     return std.fs.path.dirname(@src().file) orelse ".";
// }

// test {std.testing.refAllDecls(@This());}

/// Ideas: spatial tree that maps points to nearby triangles & supports O(1) insert and removal
///        except it's a bounding-box tree because triangles have a volume...
///        see [Bounding Volume Hierarchies](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy)
///        see [R-tree](https://en.wikipedia.org/wiki/R-tree) and variants
/// Ideas: same, but with spatial grid. we may have to increase the grid density over time...
///        gridhash points. map points →  triangles. search through nearest points until find a conflict triangle
///        update
/// Ideas: Keep a rasterized version of the triangle grid (i.e. an image!) on hand at all times, with pixel labeled by triangle id!
///        then we'll immediately know which triangle we're in!
/// Ideas: even better than that, keep essentially a grid hash at a high density, but fill each bucket with triangle id if it intersects at all!
///        then we can just do a single sweep each time we add a triangle to add it to the grid.
/// Ideas: we _could_ actually increase the density of the GridHash over time. too many bins = many bins / tri. still easy to get tri from pt.
///        but too few bins = many tris / bin. very little savings.
const V2u32 = @Vector(2, u32);
const V2i32 = @Vector(2, i32);

pub const GridHash2 = struct {
    const Self = @This();
    // const Elem = u8;
    const Elem = Tri;

    // user
    nx: u16,
    ny: u16,
    nd: u16,

    // internal
    grid: []?Elem,
    offset: Pt,
    scale: Pt,
    bbox: BBox,

    // triset: std.AutoHashMap(Elem, void),
    a: Allocator,

    pub fn init(a: Allocator, pts: []Pt, nx: u16, ny: u16, nd: u16) !Self {
        var grid = try a.alloc(?Elem, nx * ny * @as(u32, @intCast(nd)));
        for (grid) |*v| v.* = null;

        const bbox = g.boundsBBox(pts);
        const offset = Pt{ bbox.x.lo, bbox.y.lo };
        const scale = Pt{ (bbox.x.hi - bbox.x.lo) / (@as(f32, @floatFromInt(nx)) - 1e-5), (bbox.y.hi - bbox.y.lo) / (@as(f32, @floatFromInt(ny)) - 1e-5) }; //

        return Self{
            .nx = nx,
            .ny = ny,
            .nd = nd,
            .grid = grid,
            .offset = offset,
            .scale = scale,
            .bbox = bbox,
            // .triset = std.AutoHashMap(Elem, void).init(a),
            .a = a,
        };
    }

    pub fn deinit(self: *Self) void {
        self.a.free(self.grid);
        // self.triset.deinit();
        self.* = undefined;
    }

    pub fn world2grid(self: Self, world_coord: Pt) V2u32 {
        const v = @floor((world_coord - self.offset) / self.scale);
        const v0 = std.math.clamp(v[0], 0, @as(f32, @floatFromInt(self.nx - 1)));
        const v1 = std.math.clamp(v[1], 0, @as(f32, @floatFromInt(self.ny - 1)));
        return .{ @as(u32, @intFromFloat(v0)), @as(u32, @intFromFloat(v1)) };
    }

    pub fn world2gridNoBounds(self: Self, world_coord: Pt) V2i32 {
        const v = @floor((world_coord - self.offset) / self.scale);
        return .{ @as(i32, @intFromFloat(v[0])), @as(i32, @intFromFloat(v[1])) };
    }

    pub fn gridCenter2World(self: Self, gridpt: V2u32) Pt {
        // const gp = @as(V2u32,gridpt);
        const gp = (pix2Vec2(gridpt) + Pt{ 0.5, 0.5 }) * self.scale + self.offset;
        return gp;
        // const gp_world = gridpt
    }

    // World coordinates
    pub fn getWc(self: Self, pt: Pt) []?Elem {
        const pt_grid = self.world2grid(pt);
        return self.get(pt_grid);
    }

    pub fn get(self: Self, pt: V2u32) []?Elem {
        const idx = self.nd * (pt[1] * self.nx + pt[0]);
        // print("pt,idx = {d},{d}\n",.{pt,idx});
        return self.grid[idx .. idx + self.nd];
    }

    pub fn add(self: Self, pt: V2u32, val: Elem) !void {
        const mem = self.get(pt);
        for (mem) |*v| {
            if (v.* == null) {
                v.* = val;
                return;
            }
            if (triEql(v.*.?, val)) return;
        }
        return error.OutOfSpace;
    }

    // double self.nd and grid size. keep triset, nx, ny same.
    pub fn doubleSizeND(self: *Self) !void {
        const new_grid = try self.a.alloc(?Elem, self.grid.len * 2);
        for (new_grid) |*v| v.* = null;
        for (self.grid, 0..) |v, i| new_grid[2 * i] = v;
        self.a.free(self.grid);
        self.grid = new_grid;
        self.nd = 2 * self.nd;
        print("DOUBLING SIZE.... \n", .{});
    }

    fn triEql(t1: Tri, t2: Tri) bool {
        if (t1[0] == t2[0] and t1[1] == t2[1] and t1[2] == t2[2]) return true;
        return false;
    }

    // World coordinates
    pub fn addWc(self: Self, pt: Pt, val: Elem) !void {
        const pt_grid = self.world2grid(pt);
        try self.add(pt_grid, val);
    }

    // Pix coordinates
    pub fn remove(self: Self, pt_grid: V2u32, val: Elem) void {
        const mem = self.get(pt_grid);
        filter(Elem, mem, val);
    }

    fn filter(comptime T: type, arr: []?T, val: T) void {
        for (arr) |*v| {
            if (v.*) |tri| {
                if (triEql(tri, val)) v.* = null;
            }
        }

        // var head:usize = 0;
        // var tail:usize = 0;
        // // var found = false;
        // while (tail<arr.len):(tail+=1) {
        //     if (arr[tail]==null) {tail+=1; continue;}
        //     if (eql(u32,&arr[tail].?,&val)) {tail+=1; continue;} // found=true;
        //     arr[head] = arr[tail];
        //     head += 1;
        //     tail += 1;
        // }
        // for (arr[head..]) |*v| v.* = null;
        // // return found;
    }

    const eql = std.mem.eql;

    // Add circle label to every box if box centroid is inside circle
    fn addCircleToGrid(self: Self, ccircle: CircleR2, label: Elem) !void {

        // if (label==null) @breakpoint();

        // get bounding box in world coordinates
        const r = @sqrt(ccircle.r2);
        // const circle = Circle{.center=ccircle.pt , .radius=r};

        const xmin = ccircle.pt[0] - r;
        const xmax = ccircle.pt[0] + r;
        const ymin = ccircle.pt[1] - r;
        const ymax = ccircle.pt[1] + r;

        // now loop over grid boxes inside circle's bbox and set pixels inside
        var xy_min = self.world2grid(.{ xmin, ymin });
        const xy_max = self.world2grid(.{ xmax, ymax }) + V2u32{ 1, 1 }; // exclusive upper bound

        // @breakpoint();

        {
            var ix: u32 = xy_min[0];
            while (ix < xy_max[0]) : (ix += 1) {
                var iy: u32 = xy_min[1];
                while (iy < xy_max[1]) : (iy += 1) {
                    // @breakpoint();
                    // print("adding {d} , {d} , {d} \n",.{ix,iy,label});
                    // print("adding {d}\n",.{V2u32{ix,iy}});
                    try self.add(.{ ix, iy }, label);
                }
            }
        }
    }

    fn removeCircleFromGrid(self: Self, ccircle: CircleR2, label: Elem) void {

        // get bounding box in world coordinates
        const r = @sqrt(ccircle.r2);
        // const circle = Circle{.center=ccircle.pt , .radius=r};

        const xmin = ccircle.pt[0] - r;
        const xmax = ccircle.pt[0] + r;
        const ymin = ccircle.pt[1] - r;
        const ymax = ccircle.pt[1] + r;

        // now loop over grid boxes inside circle's bbox and set pixels inside
        var xy_min = self.world2grid(.{ xmin, ymin });
        const xy_max = self.world2grid(.{ xmax, ymax }) + V2u32{ 1, 1 };

        // const mem = self.get(.{0,0})[0..self.triset.count() + 3];
        // print("mem: {d}\n",.{mem});

        {
            var ix: u32 = xy_min[0];
            while (ix < xy_max[0]) : (ix += 1) {
                var iy: u32 = xy_min[1];
                while (iy < xy_max[1]) : (iy += 1) {
                    // print("removing {d} , {d} , {d} \n",.{ix,iy,label});
                    self.remove(.{ ix, iy }, label);
                    // print("mem: {d}\n",.{mem});
                }
            }
        }

        // const dxy = xy_max - xy_min + V2u32{ 1, 1 };
        // const nidx = @reduce(.Mul, dxy);
        // var idx: u16 = 0;
        // while (idx < nidx) : (idx += 1) {
        //     const pix = xy_min + V2u32{ idx / dxy[1], idx % dxy[1] };
        //     if (!pixInCircle(self, pix, ccircle)) continue;
        //     const b = self.unSetPc(pix, label);
        //     _ = b;
        // }
    }

    fn pixInCircle(self: Self, pix: V2u32, circle: CircleR2) bool {
        const pix_in_world_coords = self.gridCenter2World(pix);
        // print2(@src(),"pix {d} , wc(pix) {d} , center {d}\n", .{pix , pix_in_world_coords , circle.pt});
        const delta = pix_in_world_coords - circle.pt;
        const distance = @reduce(.Add, delta * delta);
        // print2(@src(),"distance^2 {} . r^2 {} \n", .{distance, circle.r2});
        return distance < circle.r2;
    }

    // does the sorting for you
    pub fn addTri(self: *Self, _tri: Tri, pts: []const Pt) !void {
        const tri = Mesh2D.sortTri(_tri);
        const tripts = [3]Pt{ pts[tri[0]], pts[tri[1]], pts[tri[2]] };
        const circle_r2 = g.getCircumcircle2dv2(tripts);
        self.addCircleToGrid(circle_r2, tri) catch |err| switch (err) {
            error.OutOfSpace => try self.doubleSizeND(),
            else => unreachable,
        };
        // self.triset.put(tri,{}) catch unreachable; // FIXME
    }

    pub fn remTri(self: *Self, _tri: Tri, pts: []const Pt) void {
        const tri = Mesh2D.sortTri(_tri);
        const tripts = [3]Pt{ pts[tri[0]], pts[tri[1]], pts[tri[2]] };
        const circle_r2 = g.getCircumcircle2dv2(tripts);
        self.removeCircleFromGrid(circle_r2, tri);
        // _ = self.triset.remove(tri);
    }

    // fn indexOf()

    pub fn getFirstConflict(self: Self, pt: Pt, pts: []const Pt) !Tri {
        const mem = self.get(self.world2grid(pt));
        // const idx = std.mem.indexOf(?Elem, mem, &.{null}).?;
        // print("mem: {d}\n",.{mem[0..idx + 3]});

        // var count:u8 =0;
        // for (mem) |tri| {if (tri!=null) {count += 1;}}
        // print("pix : {d} , count : {d}\n", .{self.world2grid(pt) , count});

        for (mem) |tri| {
            if (tri == null) continue;
            const tripts = [3]Pt{ pts[tri.?[0]], pts[tri.?[1]], pts[tri.?[2]] };
            if (g.pointInTriangleCircumcircle2d(pt, tripts)) {
                return tri.?;
            }
        }
        unreachable;
    }
};

fn pix2Vec2(pt: Pix) Pt {
    return Pt{
        @as(f32, @floatFromInt(pt[0])),
        @as(f32, @floatFromInt(pt[1])),
    };
}

// const CircleR2 = geo.CircleR2;

// To test if a realspace line intersects a grid box we CANT just take equally spaced samples along the line,
// because we might skip a box if the intersection is very small.

// const Circle = @import("rasterizer.zig").Circle;

// To test if a circle intersects with a grid box we need to have both objects in continuous _grid_ coordinates (not world coords and not discrete).
// There are multiple cases where an intersection might happen. Four cases where the circle overlaps one of the edges, but no corners, and four more cases where it overlaps a different corner.
// We can reduce this test in the mean time to the simple case of overlap with the center of the grid.

fn vec2Pix(v: Pt) Pix {
    return Pix{
        @as(u32, @intFromFloat(v[0])),
        @as(u32, @intFromFloat(v[1])),
    };
}

test "test trigrid" {
    var alloc = std.testing.allocator;
    var pts = try alloc.alloc(Pt, 100);
    defer alloc.free(pts);
    for (pts) |*v| v.* = .{ random.float(f32), random.float(f32) };
    var gh = try GridHash2.init(alloc, pts[0..75], 10, 10, 20);
    defer gh.deinit();
    try gh.addTri(.{ 0, 50, 74 }, pts[0..75]);
    // const res0 = gh.getWc(pts[0]);
    // std.testing.expect()
    gh.remTri(.{ 0, 50, 74 }, pts[0..75]);
}

// std.mem.containsAtLeast(comptime T: type, haystack: []const T, expected_count: usize, needle: []const T)
// fn in(comptime T:type, haystack: []const T, needle:[]const T)

// // list of grid coordinates which overlap tri
// fn triToGridBoxes(grid:GridHash , tri:Tri) [][2]u32 {}

// // add tri to each grid bucket where it has some overlap (use triToGridBoxes). need geometrical test to determine buckets from tri pts.
// fn addTriToGrid(grid:GridHash , tri:Tri, tripts:[3]Vec2) !void {}

// // find tris in pt bucket, then test each for overlap. we don't need to search other grids!
// fn getOverlappingTri(grid:GridHash, pt:Vec2) !Tri {}

// // use triToGridBoxes to compute grid boxes from tris. remove bad tris from those boxes directly.
// fn removeTrisFromGrid(grid:GridHash , tris:[]Tri) !void {}

// // For Testing

// fn randomTriangles(n:u16) []Tri {}

// test "add Tris to TriangleHash" {}

// A Mesh object consisting of triangles with vertices embedded in 2D
// Edges are implied by Triangles, i.e. they can't exist by themselves.
// ArrayList(Pt) is necessary because:
//  - can shift point positions while maintaining mesh
//  - multiple points potentially at same position (temporarily)
//  - test idx for equality instead of float values
//  - save mem in Tri{} objects as one point could be in
//  - BUT this makes removal harder... to remove a point we must either
//     - remove from self.vs and shift so self.vs is compact
//     - replace in self.vs with null or some other special value
//     - ignore it in self.vs. a point is effectively removed if no Tri or Edge point to it.
// Q: removing a point should remove associated edges and triangles ?
// Q: removing a triangle doesn't automatically remove points, because triangles share points.

// All of the relationships are many-many. point → [2,n] edges, edge → 2 pts, point → [1,n] tris, tri → 3 pts, edge → [1,2] tris (in 2d), tri → 3 edges.
// This makes it tricky to define ownership rules.

// For Bower-Watson Alg we only need triangles. We want to be able to traverse triangles quickly, and add and remove them.
pub const Mesh2D = struct {
    const Self = @This();

    vs: std.ArrayList(Pt),
    ts: std.AutoHashMap(Tri, void),
    edge2tri: std.AutoHashMap(Edge, [2]?Tri), // can also keep track of edge existence

    al: Allocator,

    pub fn deinit(self: *Self) void {
        self.vs.deinit();
        // self.es.deinit();
        self.ts.deinit();
        self.edge2tri.deinit();
    }

    pub fn init(a: Allocator, pts: []const Pt) !Self {
        var s = Self{
            .al = a,
            .vs = try std.ArrayList(Pt).initCapacity(a, 100),
            .ts = std.AutoHashMap(Tri, void).init(a),
            .edge2tri = std.AutoHashMap(Edge, [2]?Tri).init(a),
        };

        // for (pts) |p| try s.vs.appendSlice
        try s.vs.appendSlice(pts);
        return s;
    }

    pub fn initRand(a: Allocator) !Self {
        var s = Self{
            .al = a,
            .vs = try std.ArrayList(Pt).initCapacity(a, 100),
            .ts = std.AutoHashMap(Tri, void).init(a),
            .edge2tri = std.AutoHashMap(Edge, [2]?Tri).init(a),
        };

        // update pts
        {
            var i: u8 = 0;
            while (i < 4) : (i += 1) {
                const fi = @as(f32, @floatFromInt(i));
                const x = 6 * @mod(fi, 2.0) + random.float(f32);
                const y = 6 * @floor(fi / 2.0) + random.float(f32);
                s.vs.appendAssumeCapacity(.{ x, y });
            }
        }

        // update tri's
        try s.ts.put(Tri{ 0, 1, 2 }, {});
        try s.ts.put(Tri{ 1, 2, 3 }, {});

        // update edge→tri map
        try s.addTri(Tri{ 0, 1, 2 });
        try s.addTri(Tri{ 1, 2, 3 });
        // try s.addTrisToEdgeMap(try s.validTris(a));

        return s;
    }

    pub fn validTris(self: Self, a: Allocator) ![]Tri {
        const tris = blk: {
            var tri_no_null = try a.alloc(Tri, self.ts.count());
            var tail: u16 = 0;
            var it = self.ts.keyIterator();
            while (it.next()) |tri| {
                tri_no_null[tail] = tri.*;
                tail += 1;
            }
            if (a.resize(tri_no_null, tail) == false) unreachable;
            break :blk tri_no_null;
        };
        return tris;
    }

    /// add point return it's index in arraylist
    pub fn addPt(self: *Self, pt: Pt) !PtIdx {
        try self.vs.append(pt);
        return @as(PtIdx, @intCast(self.vs.items.len - 1));
    }

    /// Deprecated: add triangles whose points are already there
    pub fn addTris(self: *Self, tris: []Tri) !void {
        for (tris) |tri| {
            const tri_canonical = sortTri(tri);
            assert(tri_canonical[2] < self.vs.items.len); // make sure tri is valid
            try self.ts.put(tri, {});
        }
        try self.addTrisToEdgeMap(tris);
    }

    pub fn addTri(self: *Self, tri_unsorted: Tri) !void {
        const tri = sortTri(tri_unsorted);
        assert(tri[2] < self.vs.items.len); // make sure tri is valid
        try self.ts.put(tri, {});
        try self.addTriToEdgeMap(tri);
    }

    /// Deprecated
    pub fn addTriPts(self: *Self, tri: Tri, pts: [3]Pt) !void {
        self.addPt(pts[0]);
        self.addPt(pts[1]);
        self.addPt(pts[2]);
        // self.addTris(&[_]Tri{tri});
        self.addTris(&tri);
    }

    // remove triangles. remove edges iff no triangle exists.
    pub fn removeTri(self: *Self, tri_unsorted: Tri) void {
        const tri = sortTri(tri_unsorted);
        self.removeTriFromEdgeMap(tri);
        _ = self.ts.remove(tri);
    }

    fn addTriToEdgeMap(self: *Self, tri: Tri) !void {
        // const tri_canonical = sortTri(tri);
        const a = tri[0];
        const b = tri[1];
        const c = tri[2];
        // must refer to edge verts in sorted order!
        try self.updateEdgeAddTri(.{ a, b }, tri);
        try self.updateEdgeAddTri(.{ b, c }, tri);
        try self.updateEdgeAddTri(.{ a, c }, tri);
    }

    fn removeTriFromEdgeMap(self: *Self, tri: Tri) void {
        const a = tri[0];
        const b = tri[1];
        const c = tri[2];
        self.updateEdgeRemoveTri(.{ a, b }, tri);
        self.updateEdgeRemoveTri(.{ b, c }, tri);
        self.updateEdgeRemoveTri(.{ a, c }, tri);
    }

    const eql = std.mem.eql;

    fn updateEdgeAddTri(self: *Self, _e: Edge, _tri: Tri) !void {
        const e = sortEdge(_e);
        const tri = sortTri(_tri);

        var _entry = self.edge2tri.getPtr(e);
        if (_entry == null) {
            try self.edge2tri.put(e, .{ tri, null });
            return;
        }
        var entry = _entry.?;

        if (entry[0] == null) unreachable;
        // return error.InconsistentEdgeState;

        // at least one entry must be non-null
        if (eql(u32, &entry[0].?, &tri)) return; // already exists
        if (entry[1] == null) {
            entry[1] = tri;
            return;
        } // add it
        if (eql(u32, &entry[1].?, &tri)) return; // already exists

        // map was already full. we're trying to add a tri without deleting existing ones first.

        // return error.EdgeMapFull;
        unreachable;
    }

    // we know e maps to tri already. if it's not there, throw err.
    fn updateEdgeRemoveTri(self: *Self, e: Edge, tri: Tri) void {
        // const e = sortEdge(_e);
        // const tri = sortTri(_tri);
        var _entry = self.edge2tri.getPtr(e);
        if (_entry == null) {
            // const t2 = self.edge2tri.getPtr(.{e[1],e[0]});
            // print2(@src(),"t2 ? = {any}\n",.{t2.?.*});
            print2(@src(), "edge = {any}\n", .{e});
            // return error.EdgeDoesntExist; // must already exist.
            unreachable;
        }

        var entry = _entry.?;

        // first entry can never be null
        // if (entry[0]==null) return error.InconsistentEdgeState;

        // error if null
        const tri0 = entry[0].?;
        // not null, so try matching to tri
        const b0 = eql(u32, &tri0, &tri);
        // assign value to tri1 if not null, otherwise test b0 and either succeed or err.
        const tri1 = if (entry[1]) |ent1| ent1 else {
            // tri1 is null. if tri==tri0 then remove it and return.
            if (b0) {
                _ = self.edge2tri.remove(e);
                return;
            }
            // otherwise failure. tri0!=tri and tri1==null... inconsistent edge state.
            // the tri we want to remove isn't here !
            else {
                // print2(@src(),"edge {d} , tri {d} , entry {any} \n",.{_e,tri,entry.*});
                // return error.InconsistentEdgeState;
                unreachable;
            }
        };

        // tri1 is not null. so test it vs tri
        const b1 = eql(u32, &tri1, &tri);
        // now analyze all four cases.

        // const BB = [2]bool;

        if (b0 and b1) {
            // return error.InconsistentEdgeState;
            unreachable;
        }
        if (b0 and !b1) {
            entry.* = .{ tri1, null };
            return;
        } // shift left
        if (!b0 and b1) {
            entry.* = .{ tri0, null };
            return;
        } // remove 2nd position

        if (!b0 and !b1) {
            // return error.InconsistentEdgeState;
            unreachable;
        }
    }

    /// STUB
    /// add triangles (a,b,c) for each edge (a,b) of polygon connected to centerpoint (c).
    /// should we also remove any existing triangles (a,b,x) ?
    pub fn addPointInPolygon(self: *Self, pt_idx: PtIdx, polygon: []PtIdx) !void {
        var tri_list = try self.al.alloc(Tri, polygon.len);
        defer self.al.free(tri_list);
        for (polygon, 0..) |_, i| {
            const edge = Edge{ polygon[i], polygon[(i + 1) % polygon.len] };
            tri_list[i] = sortTri(Tri{ edge[0], edge[1], pt_idx });
        }

        // now add them to mesh and update self
        // try self.addTris(tri_list);

        // print2(@src(),"Show for pt_idx {d} poly {d} \n", .{ pt_idx, polygon });
        // self.show();
    }

    fn swap(comptime T: type, a: *T, b: *T) void {
        const temp = a.*;
        a.* = b.*;
        b.* = temp;
    }

    pub fn sortTri(_tri: Tri) Tri {
        var tri = _tri;
        if (tri[0] > tri[1]) swap(u32, &tri[0], &tri[1]);
        if (tri[1] > tri[2]) swap(u32, &tri[1], &tri[2]);
        if (tri[0] > tri[1]) swap(u32, &tri[0], &tri[1]);
        return tri;
    }

    pub fn sortEdge(edge: Edge) Edge {
        var e = edge;
        if (e[0] > e[1]) swap(PtIdx, &e[0], &e[1]);
        return e;
    }

    pub fn show(self: Self) void {
        print2(@src(), "Triangles \n", .{});

        var it_ts = self.ts.keyIterator();
        while (it_ts.next()) |tri| {
            print2(@src(), "{d} \n", .{tri.*});
        }

        print2(@src(), "Edge map \n", .{});
        var it_es = self.edge2tri.iterator();
        while (it_es.next()) |kv| {
            print2(@src(), "{d} → {d} \n", .{ kv.key_ptr.*, kv.value_ptr.* });
        }
    }

    /// up to 3 neibs ? any of them can be null;
    pub fn getTriNeibs(self: Self, tri: Tri) [3]?Tri {
        const tri_canonical = sortTri(tri);
        const a = tri_canonical[0];
        const b = tri_canonical[1];
        const c = tri_canonical[2];
        var res: [3]?Tri = undefined;
        res[0] = self.getSingleNeib(tri_canonical, .{ a, b });
        res[1] = self.getSingleNeib(tri_canonical, .{ b, c });
        res[2] = self.getSingleNeib(tri_canonical, .{ a, c });
        return res;
    }

    pub fn getSingleNeib(self: Self, tri: Tri, edge: Edge) ?Tri {
        const m2tris = self.edge2tri.get(edge);
        if (m2tris == null) return null;
        if (eql(u32, &tri, &m2tris.?[0].?)) return m2tris.?[1];
        return m2tris.?[0];
    }

    pub const raserized_width = 2500;

    pub fn rasterize(self: Self, name: []const u8) !void {
        var pix = try im.Img2D([4]u8).init(raserized_width, raserized_width);
        defer pix.deinit();

        const bbox = g.boundsBBox(self.vs.items);
        const bbox_target = g.newBBox(10, raserized_width - 10, 10, raserized_width - 10);

        for (self.vs.items) |p| {
            const p2 = g.pt2PixCast(g.affine(p, bbox, bbox_target)); // transformed
            im.drawCircle([4]u8, pix, p2[0], p2[1], 3, .{ 255, 255, 255, 255 });
        }

        var it = self.ts.keyIterator();

        while (it.next()) |_tri| {
            const tri = _tri.*;

            const p0 = g.pt2PixCast(g.affine(self.vs.items[tri[0]], bbox, bbox_target));
            const p1 = g.pt2PixCast(g.affine(self.vs.items[tri[1]], bbox, bbox_target));
            const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

            im.drawLine(
                [4]u8,
                pix,
                p0[0],
                p0[1],
                p1[0],
                p1[1],
                .{ 255, 0, 0, 255 },
            );
            im.drawLine(
                [4]u8,
                pix,
                p0[0],
                p0[1],
                p2[0],
                p2[1],
                .{ 255, 0, 0, 255 },
            );
            im.drawLine(
                [4]u8,
                pix,
                p1[0],
                p1[1],
                p2[0],
                p2[1],
                .{ 255, 0, 0, 255 },
            );
        }

        try im.saveRGBA(pix, name);
    }

    pub fn rasterizeHighlightTri(self: Self, name: []const u8, tris: [][3]u32) !void {
        var pix = try im.Img2D([4]u8).init(raserized_width, raserized_width);
        defer pix.deinit();

        const bbox = g.boundsBBox(self.vs.items);
        const bbox_target = g.newBBox(10, raserized_width - 10, 10, raserized_width - 10);

        for (self.vs.items) |p| {
            const p2 = g.pt2PixCast(g.affine(p, bbox, bbox_target)); // transformed
            im.drawCircle([4]u8, pix, p2[0], p2[1], 3, .{ 255, 255, 255, 255 });
        }

        var it = self.ts.keyIterator();

        while (it.next()) |_tri| {
            const tri = _tri.*;

            const p0 = g.pt2PixCast(g.affine(self.vs.items[tri[0]], bbox, bbox_target));
            const p1 = g.pt2PixCast(g.affine(self.vs.items[tri[1]], bbox, bbox_target));
            const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

            im.drawLine(
                [4]u8,
                pix,
                p0[0],
                p0[1],
                p1[0],
                p1[1],
                .{ 255, 0, 0, 255 },
            );
            im.drawLine(
                [4]u8,
                pix,
                p0[0],
                p0[1],
                p2[0],
                p2[1],
                .{ 255, 0, 0, 255 },
            );
            im.drawLine(
                [4]u8,
                pix,
                p1[0],
                p1[1],
                p2[0],
                p2[1],
                .{ 255, 0, 0, 255 },
            );
        }

        for (tris) |tri| {

            // const tri = tris.*;

            const p0 = g.pt2PixCast(g.affine(self.vs.items[tri[0]], bbox, bbox_target));
            const p1 = g.pt2PixCast(g.affine(self.vs.items[tri[1]], bbox, bbox_target));
            const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

            im.drawLine(
                [4]u8,
                pix,
                p0[0],
                p0[1],
                p1[0],
                p1[1],
                .{ 0, 0, 255, 255 },
            );
            im.drawLine(
                [4]u8,
                pix,
                p0[0],
                p0[1],
                p2[0],
                p2[1],
                .{ 0, 0, 255, 255 },
            );
            im.drawLine(
                [4]u8,
                pix,
                p1[0],
                p1[1],
                p2[0],
                p2[1],
                .{ 0, 0, 255, 255 },
            );
        }

        try im.saveRGBA(pix, name);
    }

    pub fn rasterizeHighlightStuff(self: Self, name: []const u8, pts: []Pt, edges: [][2]u32, tris: [][3]u32) !void {
        var pix = try im.Img2D([4]u8).init(raserized_width, raserized_width);
        defer pix.deinit();

        const bbox = g.boundsBBox(self.vs.items);
        const bbox_target = g.newBBox(10, raserized_width - 10, 10, raserized_width - 10);

        for (self.vs.items) |p| {
            const p2 = g.pt2PixCast(g.affine(p, bbox, bbox_target)); // transformed
            im.drawCircle([4]u8, pix, p2[0], p2[1], 3, .{ 255, 255, 255, 255 });
        }

        var it = self.ts.keyIterator();

        while (it.next()) |_tri| {
            const tri = _tri.*;

            const p0 = g.pt2PixCast(g.affine(self.vs.items[tri[0]], bbox, bbox_target));
            const p1 = g.pt2PixCast(g.affine(self.vs.items[tri[1]], bbox, bbox_target));
            const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

            im.drawLine(
                [4]u8,
                pix,
                p0[0],
                p0[1],
                p1[0],
                p1[1],
                .{ 255, 0, 0, 255 },
            );
            im.drawLine(
                [4]u8,
                pix,
                p0[0],
                p0[1],
                p2[0],
                p2[1],
                .{ 255, 0, 0, 255 },
            );
            im.drawLine(
                [4]u8,
                pix,
                p1[0],
                p1[1],
                p2[0],
                p2[1],
                .{ 255, 0, 0, 255 },
            );
        }

        for (tris) |tri| {
            const p0 = g.pt2PixCast(g.affine(self.vs.items[tri[0]], bbox, bbox_target));
            const p1 = g.pt2PixCast(g.affine(self.vs.items[tri[1]], bbox, bbox_target));
            const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

            im.drawLine(
                [4]u8,
                pix,
                p0[0],
                p0[1],
                p1[0],
                p1[1],
                .{ 0, 0, 255, 255 },
            );
            im.drawLine(
                [4]u8,
                pix,
                p0[0],
                p0[1],
                p2[0],
                p2[1],
                .{ 0, 0, 255, 255 },
            );
            im.drawLine(
                [4]u8,
                pix,
                p1[0],
                p1[1],
                p2[0],
                p2[1],
                .{ 0, 0, 255, 255 },
            );
        }

        for (edges) |edge| {
            const p0 = g.pt2PixCast(g.affine(self.vs.items[edge[0]], bbox, bbox_target));
            const p1 = g.pt2PixCast(g.affine(self.vs.items[edge[1]], bbox, bbox_target));
            // const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

            im.drawLine(
                [4]u8,
                pix,
                p0[0],
                p0[1],
                p1[0],
                p1[1],
                .{ 0, 255, 0, 255 },
            );
        }

        for (pts) |p| {
            const p0 = g.pt2PixCast(g.affine(p, bbox, bbox_target));
            im.drawCircle([4]u8, pix, p0[0], p0[1], 3, .{ 255, 0, 0, 255 });
        }

        try im.saveRGBA(pix, name);
    }

    // pub fn walk(self:Self, start:Tri)

    // pub fn remTri () {}

    // pub fn getTrisFromEdge(e:Edge) [2]?Tri {}

    // pub fn triExistsQ(tri:Tri) bool {}

    // pub fn walkFaces(start:Tri) []Tri {}

    // pub fn walkEdges(start:Edge) ?
};

// pub fn main() !void {
test "test rasterize()" {
    var a = std.testing.allocator;
    var the_mesh = try Mesh2D.initRand(a);
    defer the_mesh.deinit();

    the_mesh.show();
    try the_mesh.rasterize(test_home ++ "mesh0.tga");

    const tris = try the_mesh.validTris(a);
    defer a.free(tris);

    the_mesh.removeTri(tris[0]);
    try the_mesh.rasterize(test_home ++ "mesh1.tga");
    the_mesh.show();

    const center_point = (the_mesh.vs.items[tris[1][0]] + the_mesh.vs.items[tris[1][1]] + the_mesh.vs.items[tris[1][2]]) / Pt{ 3.0, 3.0 };

    _ = try the_mesh.addPt(center_point);
    _ = try the_mesh.addTri(tris[1]);
    // try the_mesh.addPointInPolygon(center_point, &tris[1]);
    the_mesh.show();
    try the_mesh.rasterize(test_home ++ "mesh2.tga");
}
