// Assume input points are in continuous grid coordinates, but haven't been clipped or rasterized.
// All funcs accept closures ([2]u32 , context) → void which are executed once per pixel.
// Are these funcs inlined?

// We want versions which trace object boundaries, but also version which fill in the object bulk.
// We want different funcs for lines, curves, circles, etc.

const std = @import("std");
const im = @import("imageBase.zig");
const cc = @import("c.zig");
const geo = @import("geometry.zig");
const cam3D = @import("Cam3D.zig");

const Img2D = im.Img2D;
const Mesh = geo.Mesh;
const Vec2 = geo.Vec2;
const Vec3 = geo.Vec3;
const PerspectiveCamera = cam3D.PerspectiveCamera;
const Mesh2D = geo.Mesh2D;

const sphereTrajectory = geo.sphereTrajectory;
const rotate2cam = cam3D.rotate2cam;
const bounds3 = geo.bounds3;
const gridMesh = geo.gridMesh;
const vec2 = geo.vec2;
const abs = geo.abs;
const uvec2 = geo.uvec2;

const min3 = std.math.min3;
const max3 = std.math.max3;
const print = std.debug.print;
const assert = std.debug.assert;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var allocator = gpa.allocator();

const test_home = "/Users/broaddus/Desktop/work/zig-tracker/test-artifacts/rasterize_continuous/";

pub inline fn inbounds(img: anytype, px: anytype) bool {
    if (0 <= px[0] and px[0] < img.nx and 0 <= px[1] and px[1] < img.ny) return true else return false;
}

const Pt = Vec2;
const Pix = @Vector(2, u32);

fn pt2Pix(x: Pt) Pix {
    return .{ @floatToInt(u32, std.math.max(0, x[0])), @floatToInt(u32, std.math.max(0, x[1])) }; // rounds down automatically ?
}

fn norm(x: Pt) Pt {
    const l = @sqrt(@reduce(.Add, x * x));
    return x / Pt{ l, l };
}

fn length(x: Pt) f32 {
    return @sqrt(@reduce(.Add, x * x));
}

pub const Ray = struct { start: Pt, stop: Pt };

/// line_segment uses continuous pix coords with pixel centers at {i+0.5,j+0.5} for i,j in [0..]
pub fn traceLineSegment(ctx: anytype, fnToRun: fn (ctx: @TypeOf(ctx), pix: Vec2) void, line_segment: Ray) void {
    var currentPoint = line_segment.start;
    const dp = line_segment.stop - line_segment.start;
    const mag = length(dp);
    const dpn = dp / Vec2{ mag, mag };

    fnToRun(ctx, currentPoint);
    var countf32: f32 = 0;

    // @breakpoint();
    while (countf32 < mag) : (countf32 += 1) {
        currentPoint += dpn;
        fnToRun(ctx, currentPoint);
        // currentPix = pt2Pix(currentPoint);
        // if (@reduce(.Or, oldPix != currentPix)) fnToRun(ctx, currentPix);
        // if (@reduce(.And, endPix == currentPix)) break;
    }
}

// TODO fill in triangles.
// TODO fill in circle.
// TODO interface that passes pix AND ~distance~ small vec to pix. We could do some interesting coulor work based on this vec!

pub const Circle = struct { center: Vec2, radius: f32 };

/// circle uses continuous pix coords with pixel centers at {i+0.5,j+0.5} for i,j in [0..]
pub fn traceCircleOutline(ctx: anytype, fnToRun: fn (ctx: @TypeOf(ctx), pix: Vec2) void, circle: Circle) void {
    const d_theta: f32 = 1 / circle.radius; // in radians 2pi / (2pi r)
    const rr = Vec2{ circle.radius, circle.radius };

    var currentAngle: f32 = 0;
    var currentPoint = rr * Vec2{ @cos(currentAngle), @sin(currentAngle) } + circle.center;
    // var currentPix = pt2Pix(currentPoint);
    fnToRun(ctx, currentPoint);

    while (true) {
        currentAngle += d_theta;
        currentPoint = rr * Vec2{ @cos(currentAngle), @sin(currentAngle) } + circle.center;
        // currentPix = pt2Pix(currentPoint);
        fnToRun(ctx, currentPoint);
        // if (@reduce(.Or,  oldPix != currentPix)) fnToRun(ctx, currentPix);
        if (currentAngle > 2 * 3.14159) break;
    }
}

// TESTING

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

const BGRA = @Vector(4, u8);
const ImgCtx = struct {
    img: Img2D([4]u8),
    val: BGRA,
};

fn fnSetValImg(ctx: ImgCtx, pt: Pt) void {
    const buf = ctx.img;
    const root2 = @sqrt(2.0);

    if (pt[0] > @intToFloat(f32, buf.nx - 1) or pt[1] > @intToFloat(f32, buf.ny - 1)) return;

    var x0 = @floor(pt[0]);
    var y0 = @floor(pt[1]);
    var err = Pt{ x0, y0 } - pt; // always magnitude < 1
    var mag = length(err);
    var color = @floatToInt(u8, (root2 - mag) * 128);
    var idx = @floatToInt(u32, y0) * buf.nx + @floatToInt(u32, x0);

    if (buf.img[idx][0] < color) buf.img[idx] = .{ color, color, color, 255 };

    if (y0 < @intToFloat(f32, buf.ny - 1)) {
        x0 = @floor(pt[0]);
        y0 = @ceil(pt[1]);
        err = Pt{ x0, y0 } - pt; // always magnitude < 1
        mag = length(err);
        color = @floatToInt(u8, (root2 - mag) * 128);
        idx = @floatToInt(u32, y0) * buf.nx + @floatToInt(u32, x0);

        if (buf.img[idx][0] < color) buf.img[idx] = .{ color, color, color, 255 };
    }

    if (x0 < @intToFloat(f32, buf.nx - 1) and y0 < @intToFloat(f32, buf.ny - 1)) {
        x0 = @ceil(pt[0]);
        y0 = @ceil(pt[1]);
        err = Pt{ x0, y0 } - pt; // always magnitude < 1
        mag = length(err);
        color = @floatToInt(u8, (root2 - mag) * 128);
        idx = @floatToInt(u32, y0) * buf.nx + @floatToInt(u32, x0);
        if (buf.img[idx][0] < color) buf.img[idx] = .{ color, color, color, 255 };
    }

    if (x0 < @intToFloat(f32, buf.nx - 1)) {
        x0 = @ceil(pt[0]);
        y0 = @floor(pt[1]);
        err = Pt{ x0, y0 } - pt; // always magnitude < 1
        mag = length(err);
        color = @floatToInt(u8, (root2 - mag) * 128);
        idx = @floatToInt(u32, y0) * buf.nx + @floatToInt(u32, x0);
        if (buf.img[idx][0] < color) buf.img[idx] = .{ color, color, color, 255 };
    }

    // if (pix[0] < 0 or pix[0] >= buf.nx or pix[1] < 0 or pix[1] >= buf.ny) return;
    im.saveRGBA(ctx.img, test_home ++ "fnSetValImg.tga") catch unreachable;
}

test "draw two lines" {
    // pub fn main() !void {
    var pic = try Img2D([4]u8).init(100, 100);
    defer pic.deinit();
    for (pic.img) |*v| v.* = .{ 0, 0, 0, 255 };

    {
        const line_segment = Ray{ .start = .{ 0.5, 0.5 }, .stop = .{ 99.99, 99.99 } };
        const ctx: ImgCtx = .{ .img = pic, .val = .{ 128, 128, 0, 255 } };
        traceLineSegment(ctx, fnSetValImg, line_segment);
    }

    {
        const line_segment = Ray{ .start = .{ 0.5, 99.99 }, .stop = .{ 99.99, 0.5 } };
        const ctx: ImgCtx = .{ .img = pic, .val = .{ 64, 255, 0, 255 } };
        traceLineSegment(ctx, fnSetValImg, line_segment);
    }

    try im.saveRGBA(pic, test_home ++ "two_lines.tga");
}

test "test traceLineSegment" {
    var img = try allocator.alloc(u8, 10 * 10);
    defer allocator.free(img);
    for (img) |*v| v.* = 0;

    const line_segment = Ray{ .start = .{ 0, 0 }, .stop = .{ 9, 9 } };
    const ctx: MyCtx = .{ .img = img, .val = 3 };
    traceLineSegment(ctx, fnSetVal, line_segment);

    print("\n", .{});
    for (img) |v, i| {
        if (i % 10 == 0) print("\n", .{});
        print("{} ", .{v});
    }

    print("\n", .{});
}

const MyCtx = struct {
    img: []u8,
    val: u8,
};
fn fnSetVal(ctx: MyCtx, pt: Pt) void {
    const x = @floatToInt(u32, pt[0]);
    const y = @floatToInt(u32, pt[1]);
    ctx.img[y * 10 + x] = ctx.val;
}
