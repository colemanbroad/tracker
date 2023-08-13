///  GridHash stores an affine mapping between continuous 2D coords and integer 2D coords used as index
/// into a storage array.
/// TODO: How to decide nx,ny,nd based on the distribution of points?
///
const std = @import("std");
const im = @import("image_base.zig");
const geo = @import("geometry.zig");

const trace = @import("trace");

pub fn log(
    comptime message_level: std.log.Level,
    comptime scope: @Type(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    _ = scope;
    _ = message_level;
    const logfile = std.fs.cwd().createFile("trace.grid_hash2.csv", .{ .truncate = false }) catch {
        std.debug.print(format, args);
        return;
    };
    logfile.seekFromEnd(0) catch {};
    logfile.writer().print(format, args) catch {};
    logfile.writer().writeByte(std.ascii.control_code.lf) catch {};
    logfile.close();
}

const print = std.debug.print;
const assert = std.debug.assert;
const expect = std.testing.expect;
const allocator = std.testing.allocator;
var arena = std.heap.ArenaAllocator.init(allocator);
var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

const Vec2 = geo.Vec2;
const BBox = geo.BBox;
// const Range = struct { hi: f32, lo: f32 };
// const BBox = struct { x: Range, y: Range };
// const RangeU32 = struct { hi: u32, lo: u32 };
// const BBoxU32 = struct { x: RangeU32, y: RangeU32 };

const CircleR2 = geo.CircleR2;

const V2u32 = @Vector(2, u32);
const V2i32 = @Vector(2, i32);

const clipi = geo.clipi;
const floor = std.math.floor;

const test_home = "/Users/broaddus/Desktop/work/isbi/zig-tracker/test-artifacts/grid_hash2/";
// const test_home = "../test-artifacts/grid_hash2/";

pub const GridHash2 = struct {
    const Self = @This();

    const Elem = u32;

    alloc: std.mem.Allocator,
    nx: u16,
    ny: u16,
    nd: u8,
    grid: []?Elem,
    offset: Vec2,
    scale: Vec2,
    bbox: BBox,

    /// stores index into pts in grid. relies on pts order!
    pub fn init(a: std.mem.Allocator, pts: []Vec2, nx: u16, ny: u16, nd: u8) !Self {
        var grid = try a.alloc(?Elem, nx * ny * nd);
        for (grid) |*v| v.* = null;

        var bbox = geo.boundsBBox(pts); // bbox for pts

        bbox.x.lo = @floor(bbox.x.lo);
        bbox.x.hi = @ceil(bbox.x.hi);
        bbox.y.lo = @floor(bbox.y.lo);
        bbox.y.hi = @ceil(bbox.y.hi);

        const offset = Vec2{ bbox.x.lo, bbox.y.lo };
        const scale = Vec2{
            (bbox.x.hi - bbox.x.lo) / (@as(f32, @floatFromInt(nx)) - 1e-5),
            (bbox.y.hi - bbox.y.lo) / (@as(f32, @floatFromInt(ny)) - 1e-5),
        }; //

        print("bbox = {}\n", .{bbox});

        var self = Self{
            .alloc = a,
            .nx = nx,
            .ny = ny,
            .nd = nd,
            .grid = grid,
            .offset = offset,
            .scale = scale,
            .bbox = bbox,
        };

        // actually add points to grid!
        for (pts, 0..) |p, i| try self.setWc(p, @as(u32, @intCast(i)));

        return self;
    }

    pub fn deinit(self: Self) void {
        self.alloc.free(self.grid);
    }

    pub fn world2grid(self: Self, world_coord: Vec2) V2u32 {
        const v = @floor((world_coord - self.offset) / self.scale);
        const v0 = std.math.clamp(v[0], 0, @as(f32, @floatFromInt(self.nx - 1)));
        const v1 = std.math.clamp(v[1], 0, @as(f32, @floatFromInt(self.ny - 1)));
        return .{ @as(u32, @intFromFloat(v0)), @as(u32, @intFromFloat(v1)) };
    }

    /// #unused
    pub fn world2gridNoBounds(self: Self, world_coord: Vec2) V2i32 {
        const v = @floor((world_coord - self.offset) / self.scale);
        return .{ @as(i32, @intFromFloat(v[0])), @as(i32, @intFromFloat(v[1])) };
    }

    /// get continuous world coordinate associated with _center_ of grid box
    pub fn gridCenter2World(self: Self, gridpt: V2u32) Vec2 {
        // const gp = @as(V2u32,gridpt);
        const gp = (pix2Vec2(gridpt) + Vec2{ 0.5, 0.5 }) * self.scale + self.offset;
        return gp;
        // const gp_world = gridpt
    }

    /// World coordinates
    pub fn getWc(self: Self, pt: Vec2) []?Elem {
        const pt_grid = self.world2grid(pt);
        const idx = self.nd * (pt_grid[0] * self.ny + pt_grid[1]);
        const res = self.grid[idx .. idx + self.nd];
        return res;
    }

    /// World coordinates
    pub fn setWc(self: Self, pt: Vec2, val: Elem) !void {
        const res = self.getWc(pt);

        for (res) |*v| {
            if (v.* == val) return;
            if (v.* == null) {
                v.* = val;
                return;
            }
        }
        return error.OutOfSpace;
    }

    /// Pix coordinates
    pub fn getPc(self: Self, pt_grid: [2]u32) []?Elem {
        const idx = self.nd * (pt_grid[0] * self.ny + pt_grid[1]);
        const res = self.grid[idx .. idx + self.nd];
        return res;
    }

    /// Pix coordinates
    pub fn setPc(self: Self, pt_grid: [2]u32, val: Elem) !void {
        const res = self.getPc(pt_grid);

        for (res) |*v| {
            if (v.* == null) {
                v.* = val;
                return;
            }
            if (v.*.? == val) return;
        }
        return error.OutOfSpace;
    }

    // Pix coordinates
    pub fn unSetPc(self: Self, pt_grid: [2]u32, val: Elem) bool {
        const res = self.getPc(pt_grid);

        for (res) |*v| {
            if (v.* == null) return false;
            if (v.*.? == val) {
                v.* = null;
                return true;
            }
        }
        return false;
    }

    /// STUB.
    /// Search neighbouring grid boxes, nearest-first, for k-nearest-neib pts.
    /// If the the radius
    /// of circle to k-1 neib overlaps with neighbouring boxes.
    /// We do circle-box intersection test to determine which grid boxes
    /// need searching.
    // pub fn getKnnWc(self: Self, pt: Vec2, k: u8) []?Elem {
    //     _ = k;
    //     _ = pt;
    // }

    // Deprecated
    // Add circle label to every box if box centroid is inside circle
    // TODO: make new datastructure for point-to-shape mappings
    pub fn addCircleToGrid(self: Self, ccircle: CircleR2, label: Elem) !void {

        // get bounding box in world coordinates
        const r = @sqrt(ccircle.r2);
        // const circle = Circle{.center=ccircle.pt , .radius=r};

        const xmin = ccircle.pt[0] - r;
        const xmax = ccircle.pt[0] + r;
        const ymin = ccircle.pt[1] - r;
        const ymax = ccircle.pt[1] + r;

        // now loop over grid boxes inside circle's bbox and set pixels inside
        var xy_min = self.world2grid(.{ xmin, ymin });
        const xy_max = self.world2grid(.{ xmax, ymax });
        print("xy_min={d}\n", .{xy_min});
        print("xy_max={d}\n", .{xy_max});
        const dxy = xy_max - xy_min + V2u32{ 1, 1 };
        const nidx = @reduce(.Mul, dxy);
        var idx: u16 = 0;
        while (idx < nidx) : (idx += 1) {
            const pix = xy_min + V2u32{ idx / dxy[1], idx % dxy[1] };

            if (!self.pixInCircle(pix, ccircle)) continue;
            try self.setPc(pix, label);
            // if (b == false) print("pix false {d}\n", .{pix});
        }
    }

    // Deprecated #unused
    pub fn removeCircleFromGrid(self: Self, ccircle: CircleR2, label: Elem) void {

        // get bounding box in world coordinates
        const r = @sqrt(ccircle.r2);
        // const circle = Circle{.center=ccircle.pt , .radius=r};

        const xmin = ccircle.pt[0] - r;
        const xmax = ccircle.pt[0] + r;
        const ymin = ccircle.pt[1] - r;
        const ymax = ccircle.pt[1] + r;

        // now loop over grid boxes inside circle's bbox and set pixels inside
        var xy_min = self.world2grid(.{ xmin, ymin });
        const xy_max = self.world2grid(.{ xmax, ymax });
        const dxy = xy_max - xy_min + V2u32{ 1, 1 };
        const nidx = @reduce(.Mul, dxy);
        var idx: u16 = 0;
        while (idx < nidx) : (idx += 1) {
            const pix = xy_min + V2u32{ idx / dxy[1], idx % dxy[1] };

            if (!self.pixInCircle(pix, ccircle)) continue;
            const b = self.unSetPc(pix, label);
            if (b == false) print("pix false {d}\n", .{pix});
        }
    }

    /// Deprecated. Used in Deprecated funcs only.
    /// To test if a circle intersects with a grid box we need to have both objects in continuous _grid_ coordinates (not world coords and not discrete).
    /// There are multiple cases where an intersection might happen. Four cases where the circle overlaps one of the edges, but no corners, and four more cases where it overlaps a different corner.
    /// We can reduce this test in the mean time to the simple case of overlap with the center of the grid.
    fn pixInCircle(grid: Self, pix: V2u32, circle: CircleR2) bool {
        const pix_in_world_coords = grid.gridCenter2World(pix);
        // print("pix {d} , wc(pix) {d} , center {d}\n", .{pix , pix_in_world_coords , circle.pt});
        const delta = pix_in_world_coords - circle.pt;
        const distance = @reduce(.Add, delta * delta);
        // print("distance^2 {} . r^2 {} \n", .{distance, circle.r2});
        return distance < circle.r2;
    }

    fn pix2Vec2(pt: V2u32) Vec2 {
        return Vec2{
            @as(f32, @floatFromInt(pt[0])),
            @as(f32, @floatFromInt(pt[1])),
        };
    }
};

/// Deprecated. Unused. Rounding performed in GridHash.world2grid()
fn vec2Pix(v: Vec2) V2u32 {
    return V2u32{
        @as(u32, @intFromFloat(v[0])),
        @as(u32, @intFromFloat(v[1])),
    };
}

fn multiindex(al: std.mem.Allocator, comptime T: type, arr: []T, index: []?u32) ![]?T {
    var new = try al.alloc(?T, index.len);
    for (index, 0..) |idx, i| {
        if (idx == null) {
            new[i] = null;
            continue;
        }
        new[i] = arr[idx.?];
    }
    return new;
}

test "test GridHash2 deterministic" {
    defer arena.deinit();
    var al = arena.allocator();
    // var pts = [_]Vec2{ .{ 0.5, 0.5 }, .{ 1.5, 1.5 }, .{ 1.5, 1.5 }, .{ 1.5, 0.5 } };
    var pts = [_]Vec2{ .{ 0.0, 0.0 }, .{ 2, 2 }, .{ 1.5, 1.5 }, .{ 1.5, 0.5 } };

    // ┌───────────┬───────────┐
    // │  0.5,0.5  │  1.5,0.5  │
    // │   null    │   null    │
    // │   null    │   null    │
    // ├───────────┼───────────┤
    // │           │  1.5,1.5  │
    // │           │  1.5,1.5  │
    // │           │   null    │
    // └───────────┴───────────┘

    var gh = try GridHash2.init(al, &pts, 2, 2, 3);
    const sl00 = gh.getPc(.{ 0, 0 });
    const sl10 = gh.getPc(.{ 1, 0 });
    const sl01 = gh.getPc(.{ 0, 1 });
    const sl11 = gh.getPc(.{ 1, 1 });

    // print("sl00 = {?d:0.3}\n", .{try multiindex(al, Vec2, &pts, sl00)});
    // print("sl10 = {?d:0.3}\n", .{try multiindex(al, Vec2, &pts, sl10)});
    // print("sl01 = {?d:0.3}\n", .{try multiindex(al, Vec2, &pts, sl01)});
    // print("sl11 = {?d:0.3}\n", .{try multiindex(al, Vec2, &pts, sl11)});
    print("\nsl00 = {any}\n", .{try multiindex(al, Vec2, &pts, sl00)});
    // for () |v| print(" {d:03d}", .{v.?});
    print("\nsl01 = {any}\n", .{try multiindex(al, Vec2, &pts, sl01)});
    // for () |v| print(" {d:03d}", .{v.?});
    print("\nsl10 = {any}\n", .{try multiindex(al, Vec2, &pts, sl10)});
    // for () |v| print(" {d:03d}", .{v.?});
    print("\nsl11 = {any}\n", .{try multiindex(al, Vec2, &pts, sl11)});
    // for () |v| print(" {d:03d}", .{v.?});
}

// pub fn main() !void {
test "test GridHash2" {
    print("\n", .{});

    // raw data
    // var pts = [_]Vec2{ .{ 0, 0 }, .{ 1.5, 1.5 }, .{ 2.5, 3 }, .{ 3, 2.5 }, .{ 7, 7 } };
    var pts: [100]Vec2 = undefined;
    for (&pts) |*p| p.* = .{ random.float(f32) * 20, random.float(f32) * 20 };

    var grid_hash = try GridHash2.init(allocator, pts[0..], 140, 140, 2);
    defer grid_hash.deinit();

    const picture = try im.Img2D([4]u8).init(140, 140);
    defer picture.deinit();

    {
        comptime var i: u16 = 0;
        inline while (i < 3) : (i += 1) {
            const ccircle = geo.getCircumcircle2dv2(pts[4 * i .. 4 * i + 3].*);
            try grid_hash.addCircleToGrid(ccircle, 100);
        }
    }

    for (pts) |p| {
        const gc = grid_hash.world2grid(p);
        im.drawCircleOutline(picture, @as(i32, @intCast(gc[0])), @as(i32, @intCast(gc[1])), 3, .{ 255, 255, 255, 255 });
    }
    try im.saveRGBA(picture, test_home ++ "trialpicture.tga");

    // now draw filled in circles
    var idx: u32 = 0;
    while (idx < picture.nx * picture.ny) : (idx += 1) {
        const gelems = grid_hash.grid[idx * grid_hash.nd .. (idx + 1) * grid_hash.nd];
        const gel = if (gelems[0]) |g| g else continue;
        picture.img[idx] = .{ @as(u8, @intCast(gel % 255)), 0, @as(u8, @intCast(gel % 255)), 255 };
    }
    try im.saveRGBA(picture, test_home ++ "trialpicture.tga");
}

// pub const GridHash_Balanced = struct {
// };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const al = gpa.allocator();
    try init_gridhash_basic(al);
}

// The most basic kind of grid hash. Evenly spaced (anisotropic) grid lines.
// We don't know how large to mak each sector in advance. Just use a
fn init_gridhash_basic(al: std.mem.Allocator) !void {
    var pts: [100]Vec2 = undefined;
    for (&pts) |*p| p.* = .{ random.float(f32) * 20, random.float(f32) * 20 };

    // make a grid with 100x100 dimensions that match with min and max grid points
    // const mima = im.minmax(Vec2, &pts);

    _ = al;

    //
}

// The balanced gridhash determines a set of grid lines that divide points up
// evenly into bins of differing size.
fn init_gridhash_balanced(al: std.mem.Allocator, pts: []Vec2) !void {
    _ = al;

    // Sort pts along one dimension
    // comptime lessThan: fn(context:@TypeOf(context), lhs:T, rhs:T)bool

    const dim = 0;
    std.sort.sort(Vec2, pts, dim, lessThanDim);
    const N = pts.len;

    _ = N;
}

test "init_gridhash_balanced" {
    var pts: [100]Vec2 = undefined;
    for (&pts) |*p| p.* = .{ random.float(f32) * 20, random.float(f32) * 20 };
    const al = std.testing.allocator;
    // const gh = try init_gridhash_balanced(al, pts);
    const gh = try init_gridhash_basic(al);
    _ = gh;
    const p_query = Vec2{ 10, 10 };
    _ = p_query;
    // const p2 = gh.nearestNeib(p_query);
    // print("{any} \n", .{ .p2 = p2 });
}

fn lessThanDim(context_idx: u8, lhs: Vec2, rhs: Vec2) bool {
    if (lhs[context_idx] < rhs[context_idx]) return true;
    return false;
}
