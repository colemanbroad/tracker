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
const Pix = @Vector(2, u32);
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

    fn gridCenter2World(self: Self, gridpt: V2u32) Vec2 {
        // const gp = @as(V2u32,gridpt);
        const gp = (pix2Vec2(gridpt) + Vec2{ 0.5, 0.5 }) * self.scale + self.offset;
        return gp;
        // const gp_world = gridpt

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

fn pix2Vec2(pt: Pix) Vec2 {
    return Vec2{
        @intToFloat(f32, pt[0]),
        @intToFloat(f32, pt[1]),
    };
}

const CircleR2 = geo.CircleR2;

// To test if a realspace line intersects a grid box we CANT just take equally spaced samples along the line,
// because we might skip a box if the intersection is very small.

// Add circle label to every box if box centroid is inside circle
pub fn addCircleToGrid(grid_hash: GridHash2, ccircle: CircleR2, label: u8) void {

    // get bounding box in world coordinates
    const r = @sqrt(ccircle.r2);
    // const circle = Circle{.center=ccircle.pt , .radius=r};

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

        if (!pixInCircle(grid_hash, pix, ccircle)) continue;
        const b = grid_hash.setPc(pix, label);
        if (b == false) print("pix false {d}\n", .{pix});
    }
}

const Circle = @import("rasterizer.zig").Circle;

// To test if a circle intersects with a grid box we need to have both objects in continuous _grid_ coordinates (not world coords and not discrete).
// There are multiple cases where an intersection might happen. Four cases where the circle overlaps one of the edges, but no corners, and four more cases where it overlaps a different corner.
// We can reduce this test in the mean time to the simple case of overlap with the center of the grid.

fn pixInCircle(grid: GridHash2, pix: V2u32, circle: CircleR2) bool {
    const pix_in_world_coords = grid.gridCenter2World(pix);
    // print("pix {d} , wc(pix) {d} , center {d}\n", .{pix , pix_in_world_coords , circle.pt});
    const delta = pix_in_world_coords - circle.pt;
    const distance = @reduce(.Add, delta * delta);
    // print("distance^2 {} . r^2 {} \n", .{distance, circle.r2});
    return distance < circle.r2;
}

pub fn main() !void {
    print("\n", .{});

    // raw data
    // var pts = [_]Vec2{ .{ 0, 0 }, .{ 1.5, 1.5 }, .{ 2.5, 3 }, .{ 3, 2.5 }, .{ 7, 7 } };
    var pts: [100]Vec2 = undefined;
    for (pts) |*p| p.* = .{ random.float(f32) * 20, random.float(f32) * 20 };

    var grid_hash = try GridHash2.init(allocator, pts[0..], 140, 140, 2);
    defer grid_hash.deinit(allocator);

    const picture = try im.Img2D([4]u8).init(140, 140);
    defer picture.deinit();

    {
        comptime var i: u16 = 0;
        inline while (i < 3) : (i += 1) {
            const ccircle = geo.getCircumcircle2dv2(pts[4 * i .. 4 * i + 3].*);
            addCircleToGrid(grid_hash, ccircle, 100);
        }
    }

    for (pts) |p| {
        const gc = grid_hash.world2grid(p);
        draw.drawCircleOutline(picture, @intCast(i32, gc[0]), @intCast(i32, gc[1]), 3, .{ 255, 255, 255, 255 });
    }
    try im.saveRGBA(picture, "trialpicture.tga");

    // now draw filled in circles
    var idx: u32 = 0;
    while (idx < picture.nx * picture.ny) : (idx += 1) {
        const gelems = grid_hash.grid[idx * grid_hash.nd .. (idx + 1) * grid_hash.nd];
        const gel = if (gelems[0]) |g| g else continue;
        picture.img[idx] = .{ gel, 0, gel, 255 };
    }
    try im.saveRGBA(picture, "trialpicture.tga");
}

fn vec2Pix(v: Vec2) Pix {
    return Pix{
        @floatToInt(u32, v[0]),
        @floatToInt(u32, v[1]),
    };
}

const draw = @import("drawing_basic.zig");

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
