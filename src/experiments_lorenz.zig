const std = @import("std");

const print = std.debug.print;

const Vec3 = geo.Vec3;
const Allocator = std.mem.Allocator;

var allocator = std.testing.allocator;

const savedir = "/Users/broaddus/Desktop/work/zig-tracker/test-artifacts/lorenz/";

// computes derivative from current state of lorenz system
// NOTE: sig=10 rho=28 beta=8/3 give cool manifold
fn lorenzEquation(xyz: Vec3, sig: f32, rho: f32, beta: f32) Vec3 {
    const x = xyz[0];
    const y = xyz[1];
    const z = xyz[2];

    const dx = sig * (y - x);
    const dy = x * (rho - z) - y;
    const dz = x * y - beta * z;

    return Vec3{ dx, dy, dz };
}

// for building small, stack-sized arrays returned by value.
// TODO: Is there a way to use this same function for producing runtime arrays?
// Or, is there a way of evaluating the runtime version of this function at comptime and getting
// an array instead of a slice ?
fn nparange(comptime T: type, comptime low: T, comptime hi: T, comptime inc: T) blk: {
    const n = @floatToInt(u16, (hi - low) / inc + 1);
    break :blk [n]T;
} {
    const n = @floatToInt(u16, (hi - low) / inc + 1);
    var arr: [n]T = undefined;
    for (arr) |*v, i| v.* = low + @intToFloat(f32, i) * inc;
    return arr;
}

test "imageToys. spinning lorenz attractor" {
    // pub fn main() !void {

    var state0 = Vec3{ 1.0, 1.0, 1.0 };
    const dt = 0.001;
    const times = nparange(f32, 0, 40, dt);
    // var state1 = state0;

    var trajectory = try allocator.alloc(Vec3, times.len);
    defer allocator.free(trajectory);

    // simplest possible integration
    for (times) |_, i| {
        trajectory[i] = state0;
        const dv = lorenzEquation(state0, 10, 28, 8 / 3);
        state0 += dv * Vec3{ dt, dt, dt };
    }

    const pic = try Img2D([4]u8).init(800, 800);

    // Draw Lorenz Attractor in 2D in three xy,xz,yz projections
    var pts: [2 * times.len]f32 = undefined;
    for (trajectory) |v, i| {
        pts[2 * i] = v[0];
        pts[2 * i + 1] = v[1];
    }

    try draw.drawPoints2D(f32, pts[0..], "lorenzXY.tga", true);
    for (trajectory) |v, i| {
        pts[2 * i] = v[0];
        pts[2 * i + 1] = v[2];
    }
    try draw.drawPoints2D(f32, pts[0..], "lorenzXZ.tga", true);
    for (trajectory) |v, i| {
        pts[2 * i] = v[1];
        pts[2 * i + 1] = v[2];
    }
    try draw.drawPoints2D(f32, pts[0..], "lorenzYZ.tga", true);

    // focus on center of manifold 3D bounding box, define camera trajectory in 3D
    const bds = geo.bounds3(trajectory);
    const focus = (bds[1] + bds[0]) / Vec3{ 2, 2, 2 };
    const spin = sphereTrajectory();
    var name = try allocator.alloc(u8, 40);
    defer allocator.free(name);
    for (spin) |campt, j| {

        // project from 3d->2d with perspective, draw lines in 2d, then save
        var cam = try PerspectiveCamera.init(
            campt * Vec3{ 300, 300, 300 },
            focus,
            1600,
            900,
            null,
        );
        defer cam.deinit();
        cc.bres.init(&cam.screen[0], &cam.nyPixels, &cam.nxPixels);
        for (trajectory) |pt, i| {
            const px = cam.world2pix(pt);
            cc.bres.setPixel(@intCast(i32, px.x), @intCast(i32, px.y));
            if (i > 0) {
                const px0 = cam.world2pix(trajectory[i - 1]);
                cc.bres.plotLine(@intCast(i32, px0.x), @intCast(i32, px0.y), @intCast(i32, px.x), @intCast(i32, px.y));
            }
        }
        name = try std.fmt.bufPrint(name, "lorenzSpin/img{:0>4}.tga", .{j});
        try im.saveF32AsTGAGreyNormedCam(cam, name);
    }
}
