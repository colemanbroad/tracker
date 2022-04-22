const std = @import("std");
const im = @import("imageBase.zig");
const geo = @import("geometry.zig");

const print = std.debug.print;
const assert = std.debug.assert;
const expect = std.testing.expect;
var allocator = std.testing.allocator;
var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

const Allocator = std.mem.Allocator;
const Vec2 = geo.Vec2;
const Vec3 = geo.Vec3;
const Range = geo.Range;
const BBox = geo.BBox;
const Tri = @import("delaunay.zig").Tri;
const GridHash = @import("grid_hash.zig").GridHash;

const clipi = geo.clipi;
const floor = std.math.floor;

const test_home = "/Users/broaddus/Desktop/work/zig-tracker/test-artifacts/TriangleHash/";

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

const GridHash2 = struct {
    const Self = @This();
    const Elem = u8;

    nx: u16,
    ny: u16,
    nd: u8,
    grid: []?Elem,
    offset: Vec2,
    scale: Vec2,
    bbox: BBox,

    fn init(a: Allocator, pts: []Vec2, nx: u16, ny: u16, nd: u8) !Self {
        var grid = try a.alloc(?Elem, nx * ny * nd);
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
        };
    }

    fn deinit(self: Self, a: Allocator) void {
        a.free(self.grid);
    }

    fn world2grid(self: Self, world_coord: Vec2) V2u32 {
        const v = @floor((world_coord - self.offset) / self.scale);
        const v0 = std.math.clamp(v[0], 0, @intToFloat(f32, self.nx - 1));
        const v1 = std.math.clamp(v[1], 0, @intToFloat(f32, self.ny - 1));
        return .{ @floatToInt(u32, v0), @floatToInt(u32, v1) };
    }

    fn world2gridNoBounds(self: Self, world_coord: Vec2) V2i32 {
        const v = @floor((world_coord - self.offset) / self.scale);
        return .{ @floatToInt(i32, v[0]), @floatToInt(i32, v[1]) };
    }

    // World coordinates
    fn getWc(self: Self, pt: Vec2) []?Elem {
        const pt_grid = self.world2grid(pt);
        const idx = self.nd * (pt_grid[0] * self.ny + pt_grid[1]);
        const res = self.grid[idx .. idx + self.nd];
        return res;
    }

    // World coordinates
    fn setWc(self: Self, pt: Vec2, val: Elem) bool {
        const res = self.getWc(pt);

        for (res) |*v| {
            if (v.* == val) return true;
            if (v.* == null) {
                v.* = val;
                return true;
            }
        }
        return false;
    }

    // Pix coordinates
    fn getPc(self: Self, pt_grid: [2]u32) []?Elem {
        const idx = self.nd * (pt_grid[0] * self.ny + pt_grid[1]);
        const res = self.grid[idx .. idx + self.nd];
        return res;
    }

    // Pix coordinates
    fn setPc(self: Self, pt_grid: [2]u32, val: Elem) bool {
        const res = self.getPc(pt_grid);

        for (res) |*v| {
            if (v.* == val) return true;
            if (v.* == null) {
                v.* = val;
                return true;
            }
        }
        return false;
    }
};

fn floatCast(vec: V2u32) Vec2 {
    return .{ @intToFloat(f32, vec[0]), @intToFloat(f32, vec[1]) };
}

pub fn main() !void {
    print("\n", .{});

    // raw data
    var pts = [_]Vec2{ .{ 0, 0 }, .{ 2.5, 2.5 }, .{ 2.5, 3 }, .{ 3, 2.5 }, .{ 7, 7 } };
    // const tri = Tri{0,1,2};

    var grid_hash = try GridHash2.init(allocator, pts[0..], 14, 14, 2);
    defer grid_hash.deinit(allocator);

    for (pts) |p| {
        print("p,w(p) {d},{d}\n", .{ p, grid_hash.world2grid(p) });
    }

    // To test if a realspace line intersects a grid box we CANT just take equally spaced samples along the line,
    // because we might skip a box if the intersection is very small.
    // But we want to check for overlap with a circumcircle!
    const ccircle = geo.getCircumcircle2dv2(pts[1..4].*);
    print("{d:.3}\n", .{ccircle});

    // get bounding box in world coordinates
    const r = @sqrt(ccircle.r2);
    const xmin = ccircle.pt[0] - r;
    const xmax = ccircle.pt[0] + r;
    const ymin = ccircle.pt[1] - r;
    const ymax = ccircle.pt[1] + r;

    // now loop over grid boxes inside circle's bbox and set pixels inside
    var xy_min = grid_hash.world2grid(.{ xmin, ymin });
    const xy_max = grid_hash.world2grid(.{ xmax, ymax });
    print("xy_min={d}\n", .{xy_min});
    print("xy_max={d}\n", .{xy_max});
    const dxy = xy_max - xy_min + V2u32{ 1, 1 };
    const nidx = @reduce(.Mul, dxy);
    var idx: u16 = 0;
    while (idx < nidx) : (idx += 1) {
        const pix = xy_min + V2u32{ idx / dxy[1], idx % dxy[1] };
        const b = grid_hash.setPc(pix, 1);
        if (b == false) print("pix false {d}\n", .{pix});
    }

    print("p      w(p)   gh(w(p)) \n", .{});
    for (pts) |p| {
        const wp = grid_hash.world2grid(p);
        const res = grid_hash.getWc(p);
        print("{d} {d} {d}\n", .{ p, wp, res });
    }
}

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
