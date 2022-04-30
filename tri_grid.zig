const std = @import("std");
const im = @import("imageBase.zig");
const geo = @import("geometry.zig");
const draw = @import("drawing_basic.zig");

const print = std.debug.print;
const assert = std.debug.assert;
const expect = std.testing.expect;
var allocator = std.testing.allocator;
var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

const Allocator = std.mem.Allocator;
const Vec2 = geo.Vec2;
const Pix = @Vector(2, u32);
const Vec3 = geo.Vec3;
const Range = geo.Range;
const BBox = geo.BBox;
// const Tri = @import("delaunay.zig").Tri;
const Tri = [3]u32;
// const GridHash = @import("grid_hash.zig").GridHash;

const clipi = geo.clipi;
const floor = std.math.floor;

const test_home = "/Users/broaddus/Desktop/work/zig-tracker/test-artifacts/tri_grid/";

pub fn print2(
    comptime src_info: std.builtin.SourceLocation,
    comptime fmt: []const u8,
    args: anytype,
) void {
    const s1 = comptime std.fmt.comptimePrint("{s}:{d}:{d} ", .{ src_info.file, src_info.line, src_info.column });
    std.debug.print(s1[41..] ++ fmt, args);
}

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
    offset: Vec2,
    scale: Vec2,
    bbox: BBox,

    triset: std.AutoHashMap(Elem, void),

    pub fn init(a: Allocator, pts: []Vec2, nx: u16, ny: u16, nd: u16) !Self {
        var grid = try a.alloc(?Elem, nx * ny * @intCast(u32, nd));
        for (grid) |*v| v.* = null;

        const bbox = geo.boundsBBox(pts);
        const offset = Vec2{ bbox.x.lo, bbox.y.lo };
        const scale = Vec2{ (bbox.x.hi - bbox.x.lo) / (@intToFloat(f32, nx) - 1e-5), (bbox.y.hi - bbox.y.lo) / (@intToFloat(f32, ny) - 1e-5) }; //

        return Self{
            .nx = nx,
            .ny = ny,
            .nd = nd,
            .grid = grid,
            .offset = offset,
            .scale = scale,
            .bbox = bbox,
            .triset = std.AutoHashMap(Elem, void).init(a),
        };
    }

    pub fn deinit(self: *Self, a: Allocator) void {
        a.free(self.grid);
        self.triset.deinit();
        self.* = undefined;
    }

    pub fn world2grid(self: Self, world_coord: Vec2) V2u32 {
        const v = @floor((world_coord - self.offset) / self.scale);
        const v0 = std.math.clamp(v[0], 0, @intToFloat(f32, self.nx - 1));
        const v1 = std.math.clamp(v[1], 0, @intToFloat(f32, self.ny - 1));
        return .{ @floatToInt(u32, v0), @floatToInt(u32, v1) };
    }

    pub fn world2gridNoBounds(self: Self, world_coord: Vec2) V2i32 {
        const v = @floor((world_coord - self.offset) / self.scale);
        return .{ @floatToInt(i32, v[0]), @floatToInt(i32, v[1]) };
    }

    pub fn gridCenter2World(self: Self, gridpt: V2u32) Vec2 {
        // const gp = @as(V2u32,gridpt);
        const gp = (pix2Vec2(gridpt) + Vec2{ 0.5, 0.5 }) * self.scale + self.offset;
        return gp;
        // const gp_world = gridpt
    }

    // World coordinates
    pub fn getWc(self: Self, pt: Vec2) []?Elem {
        const pt_grid = self.world2grid(pt);
        return self.get(pt_grid);
    }

    pub fn get(self:Self, pt:V2u32) []?Elem {
        const idx = self.nd * (pt[1] * self.nx + pt[0]);
        // print("pt,idx = {d},{d}\n",.{pt,idx});
        return self.grid[idx .. idx + self.nd];
    }

    pub fn add(self:Self, pt:V2u32, val:Elem) !void {
        const mem = self.get(pt);
        // const T = @TypeOf(val[0]);
        for (mem) |*v| {
            if (v.* == null) {v.* = val; return;}
            // if (eql(T, &v.*.?, &val)) return;
            if (triEql(v.*.?, val)) return;
        }
        return error.OutOfSpace;
    }

    fn triEql(t1:Tri, t2:Tri) bool {
        if (t1[0]==t2[0] and t1[1]==t2[1] and t1[2]==t2[2]) return true;
        return false;
    }

    // World coordinates
    pub fn addWc(self: Self, pt: Vec2, val: Elem) !void {
        const pt_grid = self.world2grid(pt);
        try self.add(pt_grid,val);
    }

    // Pix coordinates
    pub fn remove(self: Self, pt_grid: V2u32, val: Elem) void {
        const mem = self.get(pt_grid);
        filter(Elem,mem,val);
    }

    fn filter(comptime T:type, arr:[]?T, val:T) void {

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

        // get bounding box in world coordinates
        const r = @sqrt(ccircle.r2);
        // const circle = Circle{.center=ccircle.pt , .radius=r};

        const xmin = ccircle.pt[0] - r;
        const xmax = ccircle.pt[0] + r;
        const ymin = ccircle.pt[1] - r;
        const ymax = ccircle.pt[1] + r;

        // now loop over grid boxes inside circle's bbox and set pixels inside
        var xy_min = self.world2grid(.{ xmin, ymin });
        const xy_max = self.world2grid(.{ xmax, ymax }) + V2u32{1,1}; // exclusive upper bound

        // @breakpoint();

        {var ix:u32=xy_min[0];
        while(ix<xy_max[0]):(ix+=1){
            var iy:u32=xy_min[1];
            while(iy<xy_max[1]):(iy+=1){
                // @breakpoint();
                // print("adding {d} , {d} , {d} \n",.{ix,iy,label});
                // print("adding {d}\n",.{V2u32{ix,iy}});
                try self.add(.{ix,iy}, label);
            }
        }}
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
        const xy_max = self.world2grid(.{ xmax, ymax }) + V2u32{1,1};

        // const mem = self.get(.{0,0})[0..self.triset.count() + 3];
        // print("mem: {d}\n",.{mem});

        {var ix:u32=xy_min[0];
        while(ix<xy_max[0]):(ix+=1){
            var iy:u32=xy_min[1];
            while(iy<xy_max[1]):(iy+=1){
                // print("removing {d} , {d} , {d} \n",.{ix,iy,label});
                self.remove(.{ix,iy}, label);
                // print("mem: {d}\n",.{mem});
            }
        }}

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

    pub fn addTri(self: *Self, _tri: Tri, pts: []const Vec2) !void {
        const tri = geo.Mesh2D.sortTri(_tri);
        const tripts = [3]Vec2{ pts[tri[0]], pts[tri[1]], pts[tri[2]] };
        const circle_r2 = geo.getCircumcircle2dv2(tripts);
        try self.addCircleToGrid(circle_r2, tri);
        self.triset.put(tri,{}) catch unreachable; // FIXME
    }

    pub fn remTri(self: *Self, _tri: Tri, pts: []const Vec2) void {
        const tri = geo.Mesh2D.sortTri(_tri);
        const tripts = [3]Vec2{ pts[tri[0]], pts[tri[1]], pts[tri[2]] };
        const circle_r2 = geo.getCircumcircle2dv2(tripts);
        self.removeCircleFromGrid(circle_r2, tri);
        _ = self.triset.remove(tri);
    }

    pub fn getFirstConflict(self: Self, pt:Vec2, pts: []const Vec2) !Tri {
        const mem = self.get(self.world2grid(pt));
        // print("mem: {d}\n",.{mem[0..self.triset.count() + 3]});
        for (mem) |tri| {
        // var it = self.triset.iterator();
        // while (it.next()) |kv| {
        //     const tri:?Elem = kv.key_ptr.*;
            if (tri==null) break;
            const tripts = [3]Vec2{ pts[tri.?[0]], pts[tri.?[1]], pts[tri.?[2]] };
            if (geo.pointInTriangleCircumcircle2d(pt, tripts)) return tri.?;
        }
        unreachable;
        // return error.NotFound;

        // var firsttri: ?Tri = null;
        // const gridelems = trigrid.getWc(p);
        // print2(@src(), "p {d} , gridelems = {d} \n",.{p, gridelems[0..7].*});
        // var gridelems_nonull = try allo.alloc(Tri, gridelems.len);
        // defer allo.free(gridelems_nonull);
        // {
        // var head:u8=0;
        // for (gridelems) |g| {
        //     if (g==null) continue;
        //     gridelems_nonull[head] = g.?;
        //     head += 1;
        // }
        // gridelems_nonull = allo.resize(gridelems_nonull, head).?;
        // }

        // if (gridelems_nonull.len != trigrid.triset.count()) {
        //     var it = trigrid.triset.iterator();
        //     while (it.next()) |x| {
        //         print("{d}\n",.{x.key_ptr.*});
        //     }
        //     unreachable;
        // }

        // try draw_mesh.rasterizeHighlightStuff(mesh2d, "gridelems.tga", &.{p}, &.{}, gridelems_nonull);
        // // _ = try waitForUserInput();

        // for (gridelems) |maybetri,i| {
        //     print2(@src(),"maybetri {d} , i {}\n",.{maybetri,i});
        //     // assert that we will always hit a non-null element before loop ends (because we always hit a triangle)
        //     if (maybetri == null) unreachable;
        //     // get circle associated with tri and test if it contains `p`. if yes, break;
        //     const tripts = [3]Vec2{ pts[maybetri.?[0]], pts[maybetri.?[1]], pts[maybetri.?[2]] };
        //     if (geo.pointInTriangleCircumcircle2d(p, tripts)) {
        //         firsttri = geo.Mesh2D.sortTri(maybetri.?);
        //         break;
        //     }
        // }
        // if (firsttri == null) unreachable;
    }



};

fn pix2Vec2(pt: Pix) Vec2 {
    return Vec2{
        @intToFloat(f32, pt[0]),
        @intToFloat(f32, pt[1]),
    };
}

const CircleR2 = geo.CircleR2;

// To test if a realspace line intersects a grid box we CANT just take equally spaced samples along the line,
// because we might skip a box if the intersection is very small.

const Circle = @import("rasterizer.zig").Circle;

// To test if a circle intersects with a grid box we need to have both objects in continuous _grid_ coordinates (not world coords and not discrete).
// There are multiple cases where an intersection might happen. Four cases where the circle overlaps one of the edges, but no corners, and four more cases where it overlaps a different corner.
// We can reduce this test in the mean time to the simple case of overlap with the center of the grid.

fn vec2Pix(v: Vec2) Pix {
    return Pix{
        @floatToInt(u32, v[0]),
        @floatToInt(u32, v[1]),
    };
}


// test "how can i test trigrid?" {
pub fn main() !void {

    var a = std.testing.allocator;
    var pts = try a.alloc(Vec2, 100);
    defer a.free(pts);
    for (pts) |*v| v.* = .{random.float(f32),random.float(f32)};
    var gh = try GridHash2.init(a, pts[0..75], 10, 10, 20);
    defer gh.deinit(a);
    gh.addTri(.{0,50,74}, pts[0..75]);
    // const res0 = gh.getWc(pts[0]);
    // std.testing.expect()
    gh.remTri(.{0,50,74}, pts[0..75]);

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
