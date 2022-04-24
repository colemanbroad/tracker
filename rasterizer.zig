// Assume input points are in continuous grid coordinates, but haven't been clipped or rasterized.
// All funcs accept closures ([2]u32 , context) → void which are executed once per pixel.
// Are these funcs inlined?

// We want versions which trace object boundaries, but also version which fill in the object bulk.
// We want different funcs for lines, curves, circles, etc.

//

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

const test_home = "/Users/broaddus/Desktop/work/zig-tracker/test-artifacts/rasterize/";

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

pub const Ray = struct { start: Pt, stop: Pt };

/// line_segment uses continuous pix coords with pixel centers at {i+0.5,j+0.5} for i,j in [0..]
pub fn traceLineSegment(ctx: anytype, fnToRun: fn (ctx: @TypeOf(ctx), pix: Pix) void, line_segment: Ray) void {
    var currentPoint = line_segment.start;
    const dp = norm(line_segment.stop - line_segment.start) / Vec2{ 3, 3 };
    const endPix = pt2Pix(line_segment.stop);
    var currentPix = pt2Pix(currentPoint);
    fnToRun(ctx, currentPix);
    var oldPix = currentPix;

    while (true) {
        currentPoint += dp;
        currentPix = pt2Pix(currentPoint);
        if (@reduce(.Or, oldPix != currentPix)) fnToRun(ctx, currentPix);
        if (@reduce(.And, endPix == currentPix)) break;
    }
}

// TODO fill in triangles.
// TODO fill in circle.
// TODO interface that passes pix AND ~distance~ small vec to pix. We could do some interesting coulor work based on this vec!

pub const Circle = struct { center: Vec2, radius: f32 };

/// circle uses continuous pix coords with pixel centers at {i+0.5,j+0.5} for i,j in [0..]
pub fn traceCircleOutline(ctx: anytype, fnToRun: fn (ctx: @TypeOf(ctx), pix: Pix) void, circle: Circle) void {
    const d_theta: f32 = 1 / circle.radius; // in radians 2pi / (2pi r)
    const rr = Vec2{ circle.radius, circle.radius };

    var currentAngle: f32 = 0;
    var currentPoint = rr * Vec2{ @cos(currentAngle), @sin(currentAngle) } + circle.center;
    var currentPix = pt2Pix(currentPoint);
    fnToRun(ctx, currentPix);

    while (true) {
        currentAngle += d_theta;
        currentPoint = rr * Vec2{ @cos(currentAngle), @sin(currentAngle) } + circle.center;
        currentPix = pt2Pix(currentPoint);
        fnToRun(ctx, currentPix);
        // if (@reduce(.Or,  oldPix != currentPix)) fnToRun(ctx, currentPix);
        if (currentAngle > 2 * 3.14159) break;
    }
}

// TODO: rasterization is asymmetric. single pixel created in initial call to fnToRun() should be vertical line.
pub fn traceCircleFilled(ctx: anytype, fnToRun: fn (ctx: @TypeOf(ctx), pix: Pix) void, circle: Circle) void {
    const d_theta: f32 = 1 / circle.radius; // in radians 2pi / (2pi r)
    const rr = Vec2{ circle.radius, circle.radius };

    var currentAngle: f32 = 0;
    var currentPoint = rr * Vec2{ @cos(currentAngle), @sin(currentAngle) } + circle.center;
    var currentPix = pt2Pix(currentPoint);
    fnToRun(ctx, currentPix);
    var y = currentPix[1];

    var x = currentPix[0];
    var xprev = x;

    while (true) {
        currentAngle += d_theta;
        currentPoint = rr * Vec2{ @cos(currentAngle), @sin(currentAngle) } + circle.center;
        currentPix = pt2Pix(currentPoint);
        x = currentPix[0];
        if (x != xprev) {
            y = currentPix[1];
            const y_final = @floatToInt(u32, std.math.max(1, circle.center[1] + circle.radius * @sin(-currentAngle)));
            print("{} → {}\n", .{ y, y_final });
            while (y >= y_final) : (y -= 1) {
                fnToRun(ctx, .{ x, y });
            }
        }
        // if (@reduce(.Or,  oldPix != currentPix)) fnToRun(ctx, currentPix);
        if (currentAngle > 1 * 3.14159) break;
    }
}

// TESTING

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

// pub fn main() !void {
test "test traceCircleFilled" {
    const nx = 1200;
    const ny = 1000;
    var pic = try Img2D([4]u8).init(nx, ny);
    defer pic.deinit();

    for (pic.img) |*v| v.* = .{ 0, 0, 0, 255 };

    var count: u16 = 0;
    const ctx: ImgCtx = .{ .img = pic, .val = .{ 64, 255, 0, 255 } };

    while (count < 5) : (count += 1) {
        const circle = Circle{ .center = .{ nx * random.float(f32), ny * random.float(f32) }, .radius = 5 + 50 * random.float(f32) };
        traceCircleFilled(ctx, fnSetValImg, circle);
    }

    try im.saveRGBA(pic, test_home ++ "traceCircleFilled.tga");
}


test "test traceCircleOutline" {
    const nx = 1200;
    const ny = 1000;
    var pic = try Img2D([4]u8).init(nx, ny);
    defer pic.deinit();

    for (pic.img) |*v| v.* = .{ 0, 0, 0, 255 };

    var count: u16 = 0;
    const ctx: ImgCtx = .{ .img = pic, .val = .{ 64, 255, 0, 255 } };

    while (count < 500) : (count += 1) {
        const circle = Circle{ .center = .{ nx * random.float(f32), ny * random.float(f32) }, .radius = 5 + 50 * random.float(f32) };
        traceCircleOutline(ctx, fnSetValImg, circle);
    }

    try im.saveRGBA(pic, test_home ++ "traceCircleOutline.tga");
}

const BGRA = @Vector(4, u8);
const ImgCtx = struct {
    img: Img2D([4]u8),
    val: BGRA,
};
fn fnSetValImg(ctx: ImgCtx, pix: Pix) void {
    const buf = ctx.img;
    const val2 = @intCast(u8, pix[0] % 255);
    if (pix[0] < 0 or pix[0] >= buf.nx or pix[1] < 0 or pix[1] >= buf.ny) return;
    const idx = pix[1] * buf.nx + pix[0];
    buf.img[idx] = ctx.val +% BGRA{ val2 *% 2, 0, 0, 0 };
    // im.saveRGBA(ctx.img, test_home ++ "traceCircleFilled.tga") catch unreachable;
}

test "draw two lines" {
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
fn fnSetVal(ctx: MyCtx, pix: Pix) void {
    ctx.img[pix[0] * 10 + pix[1]] = ctx.val;
}
