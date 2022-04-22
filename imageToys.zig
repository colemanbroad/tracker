const std = @import("std");
const im = @import("imageBase.zig");

const cc = @import("c.zig");
const geo = @import("geometry.zig");
const draw = @import("drawing.zig");
// const mesh = @import("mesh.zig");

// const mkdirIgnoreExists = @import("tester.zig").mkdirIgnoreExists;

const Vec2 = geo.Vec2;
const Vec3 = geo.Vec3;

const pi = 3.14159265359;

const print = std.debug.print;
const assert = std.debug.assert;
const expect = std.testing.expect;
const clamp = std.math.clamp;

const abs = geo.abs;
const Mat3x3 = geo.Mat3x3;
const Ray = geo.Ray;

const normalize = geo.normalize;
const intersectRayAABB = geo.intersectRayAABB;
const cross = geo.cross;

// var gpa = std.heap.GeneralPurposeAllocator(.{}){};
// var allocator = gpa.allocator();
pub var allocator = std.testing.allocator;
const Allocator = std.mem.Allocator;

var prng = std.rand.DefaultPrng.init(0);
const rando = prng.random();

const Vector = std.meta.Vector;
const BoxPoly = geo.BoxPoly;

const Img2D = im.Img2D;
const Img3D = im.Img3D;

// const join = std.fs.path.join;
// const test_home = try join(@import("tester.zig").test_path , "imageToys/");

const join = std.fs.path.join;
const test_home = @import("tester.zig").test_path ++ "imageToys/";

test {
    std.testing.refAllDecls(@This());
}

// Projections and Cameras and Rendering

/// Orthographic projection along z.
pub fn orthProj(allo: Allocator, comptime T: type, image: Img3D(T)) ![]T {
    const nz = image.nz;
    const ny = image.ny;
    const nx = image.nx;
    var z: u32 = 0;
    var x: u32 = 0;
    var y: u32 = 0;

    // fixed const for now
    const dim = 0;

    switch (dim) {

        // Poject over Z. Order of dimensions is Z,Y,X. So X is fast and Z is slow.
        0 => {
            var res = try allo.alloc(T, ny * nx);
            const nxy = nx * ny;

            z = 0;
            while (z < nz) : (z += 1) {
                const z2 = z * nxy;

                y = 0;
                while (y < ny) : (y += 1) {
                    const y2 = y * nx;

                    x = 0;
                    while (x < nx) : (x += 1) {
                        if (res[y2 + x] < image.img[z2 + y2 + x]) {
                            res[y2 + x] = image.img[z2 + y2 + x];
                        }
                    }
                }
            }
            return res;
        },
        else => {
            unreachable;
        },
    }
}

test "imageToys. test orthProj()" {
    var nx: u32 = 200;
    var ny: u32 = 100;
    var nz: u32 = 76;
    const nxy = nx * ny;

    // Initialize memory to 0. Set bresenham global state.
    var img = try allocator.alloc(u8, nz * ny * nx);
    defer allocator.free(img); // WARNING: we put this slice into an Img3D later. Usually we call deinit(), but here we free.
    for (img) |*v| v.* = 0;

    // Generate 100 random 3D star shapes. We include boundary conditions here! This is
    // shape dependent. Might be better to separate this out into a separate call to `clamp`.
    {
        var i: u16 = 0;
        while (i < 100) : (i += 1) {
            const x0 = 1 + @intCast(u32, rando.uintLessThan(u32, nx - 2));
            const y0 = 1 + @intCast(u32, rando.uintLessThan(u32, ny - 2));
            const z0 = 1 + @intCast(u32, rando.uintLessThan(u32, nz - 2));

            // Add markers as star
            img[z0 * nxy + y0 * nx + x0] = 255;
            img[z0 * nxy + y0 * nx + x0 - 1] = 255;
            img[z0 * nxy + y0 * nx + x0 + 1] = 255;
            img[z0 * nxy + (y0 - 1) * nx + x0] = 255;
            img[z0 * nxy + (y0 + 1) * nx + x0] = 255;
            img[(z0 - 1) * nxy + y0 * nx + x0] = 255;
            img[(z0 + 1) * nxy + y0 * nx + x0] = 255;
        }
    }

    // Now let's project the image down to 2D and save it.
    var image = Img3D(u8){ .img = img, .nx = 200, .ny = 100, .nz = 76 };
    const res = try orthProj(allocator, u8, image);
    defer allocator.free(res);
    try im.saveU8AsTGAGrey(res, 100, 200, "testOrthProj.tga");
}

// SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE
// SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE
// SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE

// in place affine transform to place verts inside (nx,ny) box with 5% margins
pub fn fitbox(verts: [][2]f32, nx: u32, ny: u32) void {
    const xborder = 0.05 * @intToFloat(f32, nx);
    const xwidth = @intToFloat(f32, nx) - 2 * xborder;
    const yborder = 0.05 * @intToFloat(f32, ny);
    const ywidth = @intToFloat(f32, ny) - 2 * yborder;

    // const mima = minmax(verts);
    const mima = geo.bounds2(verts);
    const mi = mima[0];
    const ma = mima[1];

    const xrenorm = xwidth / (ma[0] - mi[0]);
    const yrenorm = ywidth / (ma[1] - mi[1]);

    for (verts) |*v| {
        const x = (v.*[0] - mi[0]) * xrenorm + xborder;
        const y = (v.*[1] - mi[1]) * yrenorm + yborder;
        v.* = .{ x, y };
    }
}

pub fn sum(comptime n: u8, T: type, vals: [][n]T) [n]T {
    var res = [1]T{0} ** n;
    comptime var count = 0;

    for (vals) |v| {
        inline while (count < n) : (count += 1) {
            res[count] += v[count];
        }
    }
    return res;
}

// in place isotropic translate & scale to fit in [0,nx] x [0,ny] box.
pub fn fitboxiso(verts: [][2]f32, nx: u32, ny: u32) void {
    const xborder = 0.05 * @intToFloat(f32, nx);
    const xwidth = @intToFloat(f32, nx) - 2 * xborder;
    const yborder = 0.05 * @intToFloat(f32, ny);
    const ywidth = @intToFloat(f32, ny) - 2 * yborder;

    const mima = geo.bounds2(verts);
    const mi = mima[0];
    const ma = mima[1];
    const xrenorm = xwidth / (ma[0] - mi[0]);
    const yrenorm = ywidth / (ma[1] - mi[1]);

    const renorm = min(xrenorm, yrenorm);
    var mean = sum(2, f32, verts);
    mean[0] /= @intToFloat(f32, verts.len);
    mean[1] /= @intToFloat(f32, verts.len);

    for (verts) |*v| {
        const x = (v.*[0] - mean[0]) * renorm + @intToFloat(f32, nx) / 2;
        const y = (v.*[1] - mean[1]) * renorm + @intToFloat(f32, ny) / 2;
        v.* = .{ x, y };
    }
}

// IMAGE FILTERS 🌇 IMAGE FILTERS 🌇 IMAGE FILTERS 🌇 IMAGE FILTERS 🌇 IMAGE FILTERS 🌇 IMAGE FILTERS
// IMAGE FILTERS 🌇 IMAGE FILTERS 🌇 IMAGE FILTERS 🌇 IMAGE FILTERS 🌇 IMAGE FILTERS 🌇 IMAGE FILTERS
// IMAGE FILTERS 🌇 IMAGE FILTERS 🌇 IMAGE FILTERS 🌇 IMAGE FILTERS 🌇 IMAGE FILTERS 🌇 IMAGE FILTERS

// XY format . TODO: ensure inline ?
pub inline fn inbounds(img: anytype, px: anytype) bool {
    if (0 <= px[0] and px[0] < img.nx and 0 <= px[1] and px[1] < img.ny) return true else return false;
}

// Run a simple min-kernel over the image to remove noise.
fn minfilter(img: Img2D(f32)) !void {
    const nx = img.nx;
    // const ny = img.ny;
    const s = img.img; // source
    const t = try allocator.alloc(f32, s.len); // target
    defer allocator.free(t);
    const deltas = [_]Vector(2, i32){ .{ -1, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 0, -1 }, .{ 0, 0 } };

    for (s) |_, i| {
        // const i = @intCast(u32,_i);
        var mn = s[i];
        const px = Vector(2, i32){ @intCast(i32, i % nx), @intCast(i32, i / nx) };
        for (deltas) |dpx| {
            const p = px + dpx;
            const v = if (inbounds(img, p)) s[@intCast(u32, p[0]) + nx * @intCast(u32, p[1])] else 0;
            mn = min(mn, v);
        }
        t[i] = mn;
    }

    // for (s) |_,i| {
    // }
    for (img.img) |*v, i| {
        v.* = t[i];
    }
}

// Run a simple min-kernel over the image to remove noise.
fn blurfilter(img: Img2D(f32)) !void {
    const nx = img.nx;
    // const ny = img.ny;
    const s = img.img; // source
    const t = try allocator.alloc(f32, s.len); // target
    defer allocator.free(t);
    const deltas = [_]Vector(2, i32){ .{ -1, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 0, -1 }, .{ 0, 0 } };

    for (s) |_, i| {
        // const i = @intCast(u32,_i);
        var x = @as(f32, 0); //s[i];
        const px = Vector(2, i32){ @intCast(i32, i % nx), @intCast(i32, i / nx) };
        for (deltas) |dpx| {
            const p = px + dpx;
            const v = if (inbounds(img, p)) s[@intCast(u32, p[0]) + nx * @intCast(u32, p[1])] else 0;
            x += v;
        }
        t[i] = x / 5;
    }

    // for (s) |_,i| {
    // }
    for (img.img) |*v, i| {
        v.* = t[i];
    }
}

test "imageToys. bitsets" {
    // pub fn main() !void {

    var bitset = std.StaticBitSet(3000).initEmpty();
    bitset.set(0);
    bitset.set(2999);

    var it = bitset.iterator(.{});
    var n = it.next();
    while (n != null) {
        print("item: {d}\n", .{n});
        n = it.next();
    }
}

// to create a stream plot we need to integrate a bunch of points through the vector field.
// note... the form of the "perlinnoise" function is just a 3D scalar field, but if we want a
// 2D random vector field we just sample with two z values > 2 apart (uncorrelated). The integration
// of the points can be exactly as simple as with Lorenz integrator. But now we need multiple points, and shorter trajectories.
//
//
// 1. draw field's vectors
// 2. draw streamlines
// 3. movie of moving streamlines
//
// 1. use a grid initial pts
// 2. use random initial pts
// 3. use circle-pack initial pts
// 4. use grid + random deviation
test "imageToys. various stream plots" {
    // pub fn main() !void {
    var nx: u32 = 1200; //3*800/2;
    var ny: u32 = 1200; //3*800/2;
    const pic = try Img2D([4]u8).init(nx, ny);
    const pts = try allocator.alloc(Vec2, 50 * 50);
    defer allocator.free(pts);

    try mkdirIgnoreExists("streamplot");

    const dx = @as(f32, 0.1);
    const dy = @as(f32, 0.1);

    for (pts) |_, i| {
        const x = @intToFloat(f32, i % 50) * dx;
        const y = @intToFloat(f32, i / 50) * dy;
        pts[i] = Vec2{ x, y };
    }

    // Perlin Noise Vectors located at `pts`
    const vals = try allocator.alloc(Vec2, pts.len);
    defer allocator.free(vals);
    for (vals) |_, i| {
        const p = pts[i];
        const x = ImprovedPerlinNoise.noise(p[0], p[1], 10.6);
        const y = ImprovedPerlinNoise.noise(p[0], p[1], 14.4); // separated in z by 2 => uncorrelated
        vals[i] = .{ @floatCast(f32, x), @floatCast(f32, y) };
    }

    // const vals = gridnoise();
    for (pic.img) |*v| v.* = .{ 0, 0, 0, 255 };

    // Draw lines on `pic`
    // Let's rescale p and v so they fit on pic.img.
    // Note we can do an isotropic rescaling without changing the properties we care about.
    // currently pts spacing is `dx`, so we should divide by dx to get a spacing of 1 pixel.
    // The bounds are (0,0) and (5,5). So we could also rescale to fit the bounds to the image.
    // for (pts) |_,j| {

    //   const p = pts[j];
    //   const v = vals[j];
    //   const x  = @floatToInt(u31, @intToFloat(f32,ny-50)*p[0]/5 + 25 + @intToFloat(f32,nx-ny)/2);
    //   const y  = @floatToInt(u31, @intToFloat(f32,ny-50)*p[1]/5 + 25);
    //   const x2 = @floatToInt(u31, @intToFloat(f32,x) + 20*v[0] );
    //   const y2 = @floatToInt(u31, @intToFloat(f32,y) + 20*v[1] );

    //   drawLine([4]u8, pic , x , y , x2 , y2 , .{255,255,255,255});

    //   // add circle at base
    //   pic.img[x-1 + nx*y]     = .{255/2 , 255/2 , 255/2 , 255};
    //   pic.img[x+1 + nx*y]     = .{255/2 , 255/2 , 255/2 , 255};
    //   pic.img[x + nx*(y-1)]   = .{255/2 , 255/2 , 255/2 , 255};
    //   pic.img[x + nx*(y+1)]   = .{255/2 , 255/2 , 255/2 , 255};
    //   pic.img[x-1 + nx*(y-1)] = .{255/2 , 255/2 , 255/2 , 255};
    //   pic.img[x+1 + nx*(y+1)] = .{255/2 , 255/2 , 255/2 , 255};
    //   pic.img[x+1 + nx*(y-1)] = .{255/2 , 255/2 , 255/2 , 255};
    //   pic.img[x-1 + nx*(y+1)] = .{255/2 , 255/2 , 255/2 , 255};

    // }
    // try im.saveRGBA(pic,"stream1.tga"); // simple vector plot. large base.

    var name = try allocator.alloc(u8, 40);
    defer allocator.free(name);
    var time: [200]f32 = undefined;
    // for (time) |*v,i| v.* = @intToFloat(f32,i) / 100;
    const dt = Vec2{ 0.002, 0.002 };

    var ptscopy = try allocator.alloc(Vec2, pts.len);
    defer allocator.free(ptscopy);
    for (ptscopy) |*p, i| p.* = pts[i];
    var nextpts = try allocator.alloc(Vec2, pts.len);
    defer allocator.free(nextpts);
    for (nextpts) |*p, i| p.* = pts[i];

    for (time) |_, j| {
        for (pts) |_, i| {
            var pt = &ptscopy[i];
            var nextpt = &nextpts[i];

            // print("j,i = {d},{d}\n", .{j,i});
            // if (j==1 and i==6) @breakpoint();

            pt.* = nextpt.*;
            // update position
            if (pt.*[0] < 0 or pt.*[1] < 0) continue;
            const deltax = 10 * @floatCast(f32, ImprovedPerlinNoise.noise(pt.*[0], pt.*[1], 10.6));
            const deltay = 10 * @floatCast(f32, ImprovedPerlinNoise.noise(pt.*[0], pt.*[1], 14.4)); // separated in z by 2 => uncorrelated
            // print("delta = {d},{d}\n",.{deltax,deltay});
            nextpt.* += Vec2{ deltax, deltay } * dt;

            // draw centered & isotropic with borders
            // const x  = @floatToInt(u31, @intToFloat(f32,ny-50)*pt.*[0]/5 + 25 + @intToFloat(f32,nx-ny)/2);
            // const y  = @floatToInt(u31, @intToFloat(f32,ny-50)*pt.*[1]/5 + 25);

            // fit to image (maybe anisotropic)
            const x1 = @floatToInt(u31, @intToFloat(f32, nx) * pt.*[0] / 5);
            const y1 = @floatToInt(u31, @intToFloat(f32, ny) * pt.*[1] / 5);
            const x2 = @floatToInt(u31, @intToFloat(f32, nx) * nextpt.*[0] / 5);
            const y2 = @floatToInt(u31, @intToFloat(f32, ny) * nextpt.*[1] / 5);

            if (!inbounds(pic, .{ x1, y1 }) or !inbounds(pic, .{ x2, y2 })) continue;

            const v0 = RGBA.fromBGRAu8(pic.img[x1 + nx * y1]);
            const v1 = RGBA{ .r = 1, .g = 1, .b = 1, .a = 0.05 };
            const v2 = v1.mix2(v0).toBGRAu8();
            draw.drawLine([4]u8, pic, x1, y1, x2, y2, v2);
        }

        name = try std.fmt.bufPrint(name, "streamplot/img{:0>4}.tga", .{j});
        try im.saveRGBA(pic, name);
    }

    // var time:u32 = 0;
    // while (time<100) : (time+=1) {
    //   for (pts) |_,i| {
    //     const i_ = @intToFloat(f32,i);
    //   }
    // }
}

/// Everything you want to know about [color blending](https://www.w3.org/TR/compositing-1/#blending) from the WWWC.
pub const RGBA = packed struct {
    r: f32,
    g: f32,
    b: f32,
    a: f32,

    const This = @This();

    // mix low-alpha x (fg) into high-alpha y (bg)
    pub fn mix(fg: This, bg: This) This {
        const alpha = 1 - (1 - fg.a) * (1 - bg.a);
        if (alpha < 1e-6) return .{ .r = 0, .g = 0, .b = 0, .a = alpha };
        const w0 = fg.a / alpha;
        const w1 = bg.a * (1 - fg.a) / alpha;
        const red = fg.r * w0 + bg.r * w1;
        const green = fg.g * w0 + bg.g * w1;
        const blue = fg.b * w0 + bg.b * w1;
        return .{ .r = red, .g = green, .b = blue, .a = alpha };
    }

    pub fn mix2(x: This, y: This) This {
        const alpha = 1 - (1 - x.a) * (1 - y.a);
        if (alpha < 1e-6) return .{ .r = 0, .g = 0, .b = 0, .a = alpha };
        const w0 = x.a / (x.a + y.a);
        const w1 = y.a / (x.a + y.a);
        const red = x.r * w0 + y.r * w1;
        const green = x.g * w0 + y.g * w1;
        const blue = x.b * w0 + y.b * w1;
        return .{ .r = red, .g = green, .b = blue, .a = alpha };
    }

    pub fn fromBGRAu8(a: [4]u8) This {
        const v0 = @intToFloat(f32, a[0]) / 255;
        const v1 = @intToFloat(f32, a[1]) / 255;
        const v2 = @intToFloat(f32, a[2]) / 255;
        const v3 = @intToFloat(f32, a[3]) / 255;
        return .{ .r = v2, .g = v1, .b = v0, .a = v3 };
    }

    pub fn toBGRAu8(a: This) [4]u8 {
        const v0 = @floatToInt(u8, a.b * 255);
        const v1 = @floatToInt(u8, a.g * 255);
        const v2 = @floatToInt(u8, a.r * 255);
        const v3 = @floatToInt(u8, a.a * 255);
        return [4]u8{ v0, v1, v2, v3 };
    }
};

const Mesh = geo.Mesh;

test "imageToys. render soccerball with occlusion" {

    // pub fn main() !void {
    var box = BoxPoly.createAABB(.{ 3, 3, 3 }, .{ 85, 83, 84 });
    var a = box.vs.len;
    var b = box.es.len;
    var c = box.fs.len; // TODO: FIXME: I shouldn't have to do this just to convert types....
    const surf1 = Mesh{ .vs = box.vs[0..a], .es = box.es[0..b], .fs = box.fs[0..c] };
    const surf2 = try geo.subdivideMesh(surf1, 3);
    defer surf2.deinit();

    try mkdirIgnoreExists("soccerball");
    try draw.drawMesh3DMovie2(surf2, "soccerball/img");
}

test "imageToys. random scattering of points with drawPoints2D()" {
    var points2D: [100_000]f32 = undefined; // 100k f32's living on the Stack
    for (points2D) |*v| v.* = rando.float(f32);
    try draw.drawPoints2D(f32, points2D[0..], "scatter.tga", false);
}

test "imageToys. drawPoints2D() Spiral" {
    // pub fn main() !void {
    const N = 10_000;
    var points2D: [N]f32 = undefined;
    // Arrange points in spiral
    for (points2D) |*v, i| v.* = blk: {
        const _i = @intToFloat(f32, i);
        if (i % 2 == 0) {
            const x = @cos(30 * _i / N * 6.28) * 0.5 * _i / N + 0.5;
            break :blk x;
        } else {
            const y = @sin(30 * _i / N * 6.28) * 0.5 * _i / N + 0.5;
            break :blk y;
        }
    };
    try draw.drawPoints2D(f32, points2D[0..], "spiral.tga", true);
}

// DATA GENERATION  DATA GENERATION
// DATA GENERATION  DATA GENERATION
// DATA GENERATION  DATA GENERATION

pub fn random2DMesh(allo: Allocator, nx: f32, ny: f32) !struct { verts: []Vec2, edges: std.ArrayListUnmanaged([2]u32) } {
    const verts = try allo.alloc(Vec2, 100);
    var edges = try std.ArrayListUnmanaged([2]u32).initCapacity(allo, 1000); // NOTE: we must use `var` for edges here, even though it's referring to slices on the heap? how?

    for (verts) |*v, i| {
        const x = rando.float(f32) * nx;
        const y = rando.float(f32) * ny;
        v.* = .{ x, y };

        const n1 = rando.uintLessThan(u32, @intCast(u32, verts.len));
        const n2 = rando.uintLessThan(u32, @intCast(u32, verts.len));
        const n3 = rando.uintLessThan(u32, @intCast(u32, verts.len));
        edges.appendAssumeCapacity(.{ @intCast(u32, i), n1 });
        edges.appendAssumeCapacity(.{ @intCast(u32, i), n2 });
        edges.appendAssumeCapacity(.{ @intCast(u32, i), n3 });
    }
    const ret = .{ .verts = verts, .edges = edges };
    return ret;
}

test "imageToys. random mesh" {
    const pic = try Img2D([4]u8).init(800, 800);
    var msh = try random2DMesh(allocator, 800, 800);
    defer allocator.free(msh.verts);
    defer msh.edges.deinit(allocator);

    for (msh.edges.items) |e| {
        const x0 = @floatToInt(u31, msh.verts[e[0]][0]);
        const y0 = @floatToInt(u31, msh.verts[e[0]][1]);
        const x1 = @floatToInt(u31, msh.verts[e[1]][0]);
        const y1 = @floatToInt(u31, msh.verts[e[1]][1]);
        draw.drawLine([4]u8, pic, x0, y0, x1, y1, .{ 255, 0, 255, 255 });
    }
    try im.saveRGBA(pic, try join(test_home, "meshme.tga"));
}

// spiral walk around the unit sphere
pub fn sphereTrajectory() [100]Vec3 {
    var phis: [100]f32 = undefined;
    // for (phis) |*v,i| v.* = ((@intToFloat(f32,i)+1)/105) * pi;
    for (phis) |*v, i| v.* = ((@intToFloat(f32, i)) / 99) * pi;
    var thetas: [100]f32 = undefined;
    // for (thetas) |*v,i| v.* = ((@intToFloat(f32,i)+1)/105) * 2*pi;
    for (thetas) |*v, i| v.* = ((@intToFloat(f32, i)) / 99) * 2 * pi;
    var pts: [100]Vec3 = undefined;
    for (pts) |*v, i| v.* = Vec3{ @cos(phis[i]), @sin(thetas[i]) * @sin(phis[i]), @cos(thetas[i]) * @sin(phis[i]) }; // ZYX coords
    return pts;
}

test "imageToys. Perlin noise (improved)" {
    // pub fn main() !void {

    var noise = try allocator.alloc(f32, 512 * 512);
    defer allocator.free(noise);
    for (noise) |*v, i| {
        const x = @intToFloat(f64, i / 512) / 2;
        const y = @intToFloat(f64, i % 512) / 2;
        const z = 0;
        v.* = @floatCast(f32, ImprovedPerlinNoise.noise(x, y, z));
    }

    try im.saveF32AsTGAGreyNormed(noise, 512, 512, "perlinnoise.tga");
}

// Improved Perlin Noise
// Ported directly from Ken Perlin's Java implementation: https://mrl.cs.nyu.edu/~perlin/noise/
// Correlations fall off to zero at distance=2 ?
const ImprovedPerlinNoise = struct {
    pub fn noise(_x: f64, _y: f64, _z: f64) f64 {
        var x = _x; // @floor(x);                                // FIND RELATIVE X,Y,Z
        var y = _y; // @floor(y);                                // OF POINT IN CUBE.
        var z = _z; // @floor(z);

        var X = @floatToInt(u32, @floor(x)) & @as(u32, 255); // & 255;                   // FIND UNIT CUBE THAT
        var Y = @floatToInt(u32, @floor(y)) & @as(u32, 255); // & 255;                   // CONTAINS POINT.
        var Z = @floatToInt(u32, @floor(z)) & @as(u32, 255); // & 255;
        x -= @floor(x); // FIND RELATIVE X,Y,Z
        y -= @floor(y); // OF POINT IN CUBE.
        z -= @floor(z);
        var u: f64 = fade(x); // COMPUTE FADE CURVES
        var v: f64 = fade(y); // FOR EACH OF X,Y,Z.
        var w: f64 = fade(z);
        // HASH COORDINATES OF THE 8 CUBE CORNERS
        var A: u32 = p[X] + Y;
        var AA: u32 = p[A] + Z;
        var AB: u32 = p[A + 1] + Z;
        var B: u32 = p[X + 1] + Y;
        var BA: u32 = p[B] + Z;
        var BB: u32 = p[B + 1] + Z;

        return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), // AND ADD
            grad(p[BA], x - 1, y, z)), // BLENDED
            lerp(u, grad(p[AB], x, y - 1, z), // RESULTS
            grad(p[BB], x - 1, y - 1, z))), // FROM  8
            lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1), // CORNERS
            grad(p[BA + 1], x - 1, y, z - 1)), // OF CUBE
            lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))));
    }
    pub fn fade(t: f64) f64 {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }
    pub fn lerp(t: f64, a: f64, b: f64) f64 {
        return a + t * (b - a);
    }
    pub fn grad(hash: u32, x: f64, y: f64, z: f64) f64 {
        var h: u32 = hash & 15; // CONVERT LO 4 BITS OF HASH CODE
        var u: f64 = if (h < 8) x else y; // INTO 12 GRADIENT DIRECTIONS.
        var v: f64 = if (h < 4) y else if (h == 12 or h == 14) x else z;
        // return (if ((h&1) == 0) u else -u) + ((h&2) == 0 ? v : -v);
        return (if ((h & 1) == 0) u else -u) + (if ((h & 2) == 0) v else -v);
    }

    const permutation = [_]u32{ 151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180 };

    // for (int i=0; i < 256 ; i++) p[256+i] = p[i] = permutation[i];
    const p = blk: {
        var _p: [2 * permutation.len]u32 = undefined;
        var i: u32 = 0;
        while (i < 256) : (i += 1) {
            _p[i] = permutation[i];
            _p[256 + i] = permutation[i];
        }
        break :blk _p;
    };
};

pub fn gridnoise(nx: u32, ny: u32) !Img2D(Vec2) {
    const noise = try Img2D(Vec2).init(nx, ny);
    for (noise.img) |*v, i| {
        v.* = Vec2{ @intToFloat(f32, i % nx), @intToFloat(f32, i / nx) };
        v.* += Vec2{ rando.float(f32), rando.float(f32) };
    }
    return noise;
}

test "imageToys. vec syntax" {
    const a = [2]f32{ 1, 9 };
    // _ = [2]f32{4,7};
    _ = geo.vec2(a);

    const d: u32 = 100;
    const e: u32 = 200;
    const f = @intCast(i33, d) - e;
    print("\nf={}\n", .{f});
}

const max = std.math.max;
const min = std.math.min;

const u16BorderVal = ~@as(u16, 0);
const u16UnMappedLabel = u16BorderVal - 1;

pub fn fillImg2D(img: []u8, nx: u32, ny: u32) !struct { img: []u16, maxLabel: u16 } {

    // Special label values
    // u16UnMappedLabel  = initial value for unvisited pixels
    // u16BorderVal = value for non-labeled regions

    // Initial conditions...
    // We need to know which sites are borders (and can't be merged) and which sites haven't been visited yet.
    // We can make both of these values `poisonVal` initially? But leave the border values as poison after visiting them.
    var res = try allocator.alloc(u16, img.len); // return this value, don't free
    for (res) |*v| v.* = u16BorderVal; // use border val for unvisited labels as well!
    for (img) |*v, i| {
        if (v.* > 0) {
            res[i] = u16BorderVal; // non-label boundary value (always remapped to zero)
        }
    }

    // Algorithm internal state
    var currentLabel: ?u16 = null;
    var currentMax: u16 = 0;

    var dj = try DisjointSets.init(allocator, 64_000);
    defer dj.deinit();
    // var map = std.AutoHashMap(u16,u16).init(allocator);
    // defer map.deinit();

    // First loop over the image. Label each pixel greedily, but build up a hash
    // map of pixels to remap.
    // print("Stage One:\n", .{});
    {
        var i: u32 = 0;
        while (i < ny) : (i += 1) {
            var j: u32 = 0;
            while (j < nx) : (j += 1) {
                const v = img[i * nx + j];

                // if v==0 we need to draw a label
                if (v == 0) {

                    // Get set of all neib label values and their associated SetRoot.
                    // Let's do boundary conditions by assuming a Border outside the image.
                    const l1 = if (j > 0) res[i * nx + j - 1] else u16BorderVal;
                    // const l2  = if (j<nx-1) res[i*nx + j+1]   else u16BorderVal;
                    const l3 = if (i > 0) res[(i - 1) * nx + j] else u16BorderVal;
                    // const l4  = if (i<ny-1) res[(i+1)*nx + j] else u16BorderVal;

                    // root is SetRoot of all neighbour labels
                    // const root = dj.merge(l1,dj.merge(l2,dj.merge(l3,l4)));
                    const root = dj.merge(l1, l3);

                    if (currentLabel) |cl| { // both currentLabel and root are valid. merge them and use the new root.
                        currentLabel = dj.merge(cl, root);
                    } else if (root < u16UnMappedLabel) { // currentLabel is void, but root is good. use root.
                        currentLabel = root;
                    } else { // currentLabel is void and root is poison. make a new highest label and use it.
                        currentMax += 1;
                        currentLabel = currentMax;
                    }

                    res[i * nx + j] = currentLabel.?;

                    // Otherwise we don't need to draw a new label. Thus, we also
                    // don't need to add any remapping to the hashmap. But we do need to
                    // unset the currentLabel (make sure it's null).
                } else {
                    res[i * nx + j] = u16BorderVal;
                    currentLabel = null;
                }
            }
        }
    }

    // try countPixelValues(res);
    // print("Stage Two:\n", .{});

    // try map.put(u16BorderVal, 0); // remap at the end. we want the stand-in value to be high to simplify the alg.
    // dj.map[u16BorderVal] = 0;

    // const maxval = im.minmax(u16,res)[1];
    // assert(maxval < u16UnMappedLabel);

    // try map.put(u16UnMappedLabel,  0); // remap at the end. we want the stand-in value to be high to simplify the alg.
    // Now loop over every pixel and lookup the label value. Remap if needed.
    {
        var i: u32 = 0;
        while (i < ny) : (i += 1) {
            var j: u32 = 0;
            while (j < nx) : (j += 1) {
                const label = res[i * nx + j];
                if (label == u16BorderVal) {
                    res[i * nx + j] = 0;
                } else {
                    res[i * nx + j] = dj.find(label);
                }
            }
        }
    }

    const ans = .{ .img = res, .maxLabel = currentMax }; // TODO: why is this function able to infer the return type? i thought it couldn't do this?
    return ans;
}

test "imageToys. color square grid with fillImg2D()" {
    // fn fillImg2DSimple() !void {
    // print("\n", .{});

    var nx: u32 = 101;
    var ny: u32 = 101;

    // Make tic-tac-toe board image
    var img = try allocator.alloc(u32, nx * ny);
    defer allocator.free(img);
    for (img) |*v, i| {
        var ny_ = i / nx;
        var nx_ = i % nx;
        if (nx_ % 10 == 0 or ny_ % 10 == 0) v.* = 1 else v.* = 0;
    }

    // label the empty 0-valued spaces between bezier crossings
    var imgu8 = try allocator.alloc(u8, nx * ny);
    defer allocator.free(imgu8);
    for (imgu8) |*v, i| v.* = @intCast(u8, img[i]);
    const _res = try fillImg2D(imgu8, @intCast(u16, nx), @intCast(u16, ny));
    defer allocator.free(_res.img);
    const res = _res.img;
    const maxLabel = _res.maxLabel;
    print("MaxLabel = {}\n", .{maxLabel});

    // try printPixelValueCounts(res);

    // std.sort.sort(u16 , res , {} , comptime std.sort.asc(u16)); // block sort
    // print("\n{d}\n",.{res[res.len-100..]});

    // Save the result as a greyscale image.
    for (imgu8) |*v, i| v.* = @intCast(u8, res[i] % 256);
    try im.saveU8AsTGAGrey(imgu8, @intCast(u16, ny), @intCast(u16, nx), "fill_simple.tga");

    for (imgu8) |*v, i| v.* = @intCast(u8, img[i] * 255);
    try im.saveU8AsTGAGrey(imgu8, @intCast(u16, ny), @intCast(u16, nx), "fill_simple_board.tga");

    // // Print the image in a grid
    // for (res) |*v,i| {
    //   if (i%101==0) print("\n",.{});
    //   print("{d} ",.{v.*});
    // }
}

// Disjoint Set data structure. see [notes.md @ Fri Sep  3]
// Used to keep track of labels in `fillImg2D()`
const DisjointSets = struct {
    const This = @This();
    const Elem = u16;
    const SetRoot = u16;
    const poisonVal = u16BorderVal;
    const unknownVal = u16BorderVal - 1;
    const rootParent = u16BorderVal - 2;
    // labels in use are [1..rootParent-1]

    map: []Elem,
    nElems: usize,
    allo: Allocator,

    pub fn init(allo: Allocator, n: usize) !This {
        // const _map = try allocator.alloc(Node, ids.len);
        const _map = try allocator.alloc(Elem, n);
        // All elements are in their own set to start off.
        for (_map) |*v| v.* = rootParent;
        return This{ .map = _map, .nElems = n, .allo = allo };
    }

    pub fn deinit(self: This) void {
        self.allo.free(self.map);
    }

    // Double the number of labels in the total set. [Uses 👇 allocator.resize()]
    pub fn grow(this: This) !void {
        this.nElems *= 2;
        this.map = try allocator.resize(this.map, this.nElems);
        for (this.map[this.nElems / 2 ..]) |*v| v.* = rootParent;
    }
    // Return the SetRoot label associated with an arbitrary label.
    pub fn find(this: This, elem: Elem) SetRoot {
        var current = elem;
        var parent = this.map[@intCast(usize, elem)];
        while (parent != rootParent) {
            current = parent;
            parent = this.map[@intCast(usize, current)];
        }
        return current;
    }

    // Find sets associated with elements s1 and s2 and merge them together, returning the lower-valued SetRoot label.
    // Reserve poisonVal for an element that must always be in it's own singleton set and cannot be merged.
    pub fn merge(this: This, s1: Elem, s2: Elem) SetRoot {

        // return early if at least one value is invalid
        if (s1 >= rootParent and s2 < rootParent) return s2;
        if (s2 >= rootParent and s1 < rootParent) return s1;
        if (s1 >= rootParent and s2 >= rootParent) return s1;

        // else they must be valid
        const root1 = this.find(s1);
        const root2 = this.find(s2);
        if (root1 == root2) return root1;
        if (root1 < root2) {
            this.map[@intCast(usize, root2)] = root1;
            return root1;
        } else {
            this.map[@intCast(usize, root1)] = root2;
            return root2;
        }
    }

    // TODO: Implement a function to create Sets of Values (instead of returning the SetRoot). Should be more efficient than iterating find() over all keys.
};

test "imageToys. DisjointSets datastructure basics" {
    // fn testDisjointSets() !void {
    var dj = try DisjointSets.init(allocator, 1000);
    defer dj.deinit();
    print("{}\n", .{dj.find(5)});
    _ = dj.merge(5, 9);
    _ = dj.merge(3, 8);
    _ = dj.merge(3, 5);
    print("{}\n", .{dj.find(3)});
    print("{}\n", .{dj.find(5)});
    print("{}\n", .{dj.find(8)});
    print("{}\n", .{dj.find(9)});
}

// remap each element in the greyscale array to a random RGB value (returned array is 4x size)
pub fn randomColorLabels(allo: Allocator, comptime T: type, lab: []T) ![]u8 {
    const mima = im.minmax(T, lab);
    var rgbmap = try allo.alloc(u8, 3 * (@intCast(u16, mima[1]) + 1));
    defer allo.free(rgbmap);
    for (rgbmap) |*v| v.* = rando.int(u8);

    // map 0 to black
    rgbmap[0] = 0;
    rgbmap[1] = 0;
    rgbmap[2] = 0;
    rgbmap[3] = 255;

    // make new recolored image
    var rgbImg = try allo.alloc(u8, 4 * lab.len);
    for (lab) |*v, i| {
        const l = 3 * @intCast(u16, v.*);
        rgbImg[4 * i + 0] = rgbmap[l + 0];
        rgbImg[4 * i + 1] = rgbmap[l + 1];
        rgbImg[4 * i + 2] = rgbmap[l + 2];
        rgbImg[4 * i + 3] = 255;
    }

    return rgbImg;
}

// iterate through all hashmap values and print "key -> val"
fn printHashMap(map: std.AutoHashMap(u16, u16)) void {
    {
        var i: u16 = 0;
        while (i < 20) : (i += 1) {
            const x = map.get(i);
            print("{} → {}\n", .{ i, x });
        }
    }
}

pub fn printPixelValueCounts(img: []u16) !void {
    var map = std.AutoHashMap(u16, u32).init(allocator); // label -> count
    defer map.deinit();
    var maxLabel: u16 = 0;
    for (img) |*v| {
        if (v.* > maxLabel) maxLabel = v.*;
        const count = map.get(v.*);
        if (count) |ct| {
            try map.put(v.*, ct + 1);
        } else {
            try map.put(v.*, 1);
        }
    }

    // Print out the label counts
    print("histogram\n", .{});
    {
        var i: u16 = 0;
        while (i < maxLabel) : (i += 1) {
            const count = map.get(i);
            if (count) |ct| {
                if (ct > 0) print("{d} -> {d}\n", .{ i, ct });
            }
        }
    }
}

// const tracy = @import("tracylib.zig");

// 2D disk sampling. caller owns returned memory.
// pub fn random2DPoints(npts:u32) ![]Vec2 {
// pub fn main() !void {
test "imageToys. 2d stratified sampling and radial distribution function" {

    // const t = tracy.trace(@src(), null);
    // defer t.end();

    const npts: u32 = 120 * 100;
    const img = try Img2D([4]u8).init(1200, 1000);

    // clear screen
    for (img.img) |*v| v.* = .{ 0, 0, 0, 255 };

    var pts = try allocator.alloc(Vec2, npts);
    defer allocator.free(pts);

    {
        var i: u32 = 0;
        while (i < npts) : (i += 1) {
            const x = rando.uintLessThan(u31, @intCast(u31, img.nx));
            const y = rando.uintLessThan(u31, @intCast(u31, img.ny));

            pts[@intCast(u32, i)] = .{ @intToFloat(f32, x), @intToFloat(f32, y) };

            // const idx = x + y*img.nx;
            // img.img[idx] = .{128,0,0,255};
            draw.drawCircle([4]u8, img, x, y, 3, .{ 128, 128, 0, 255 });
        }
    }
    try im.saveRGBA(img, "randomptsEvenDistribution.tga");

    // clear screen
    for (img.img) |*v| v.* = .{ 0, 0, 0, 255 };

    const nx = 120;
    // const nx_f32 = @as(f32,nx);
    // const ny = 100;

    {
        var i: i32 = 0;
        while (i < npts) : (i += 1) {
            const x = @mod(i, nx) * 10 + 5;
            const y = @divFloor(i, nx) * 12 + 6;
            // print("x,y = {d} , {d}\n", .{x,y});
            const x2 = x + rando.intRangeAtMost(i32, -2, 2);
            const y2 = y + rando.intRangeAtMost(i32, -3, 3);

            // pts[@intCast(u32,i)] = .{@intToFloat(f32,x2) , @intToFloat(f32,y2)};

            // const idx = x + y*img.nx;
            // img.img[idx] = .{128,0,0,255};

            // const Myrgba = packed struct {r:u8,g:u8,b:u8,a:u8};
            var rgba = @bitCast([4]u8, @bitReverse(i32, i));
            // rgba[3] = 255;
            rgba = .{ 255, 255, 255, 255 };

            // drawCircle([4]u8, img, x2, y2, 1, .{128,128,0,255});
            draw.drawCircle([4]u8, img, x2, y2, 1, rgba);
        }
    }
    try im.saveRGBA(img, "randomptsGridWRandomDisplacement.tga");

    try radialDistribution(pts);
    // try drawPoints2D(Vec2, points2D:[]T, picname:Str, lines:?bool)
    // const floatpic = try Img2D(f32).init(1000,1000);
    // drawVerts(floatpic , verts)
}

const absInt = std.math.absInt;
// const assert = std.debug.assert;

pub fn radialDistribution(pts: []Vec2) !void {
    _ = pts.len; // num particles
    // const pts2f32 = std.mem.bytesAsSlice([2]f32, std.mem.sliceAsBytes(pts));
    // `pts` natural coordinates...
    const bounds = geo.bounds2(pts);
    const midpoint = Vec2{ (bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2 };
    const width = bounds[1][0] - bounds[0][0];
    const height = bounds[1][1] - bounds[0][1];

    const pt2pic = struct {
        pub fn f(pt: Vec2, prm: anytype) [2]u32 {
            const pt_pix = (pt - prm.m) / Vec2{ prm.h, prm.h } * Vec2{ 500, 500 } + Vec2{ 1000 * prm.r, 1000 };
            const res = .{ @floatToInt(u32, pt_pix[0]), @floatToInt(u32, pt_pix[1]) };
            return res;
        }
    }.f;

    // `count` will bin the points onto the pixel grid. Then `pic` will have a color.
    const count = try Img2D(u32).init(@floatToInt(u32, width / height * 2000), 2000);
    for (count.img) |*c| c.* = 0;

    for (pts) |pt0, i| {
        for (pts) |pt1, j| {
            if (i == j) continue;
            const prm = .{ .h = height, .m = midpoint, .r = width / height };
            const pix = pt2pic(pt1 - pt0, prm);
            count.img[pix[0] + 2000 * pix[1]] += 1;
        }
    }

    const pic = try Img2D([4]u8).init(count.nx, count.ny);
    for (pic.img) |*rgba| rgba.* = .{ 0, 0, 0, 255 };
    // pic.img[pix[0] + 2000*pix[1]] = .{255,255,255,255};
    const count_max = std.mem.max(u32, count.img);
    for (pic.img) |*rgba, i| {
        // const val = @intToFloat(f32,count.img[i]) / @intToFloat(f32,count_max);
        const v = @intCast(u8, @divFloor(count.img[i] * 255, count_max));
        rgba.* = .{ v, v, v, 255 };
    }

    try im.saveRGBA(pic, "radialDisticution.tga");
}

// USES BRESHENHAM 👇 USES BRESHENHAM 👇 USES BRESHENHAM 👇 USES BRESHENHAM 👇 USES BRESHENHAM 👇 USES BRESHENHAM
// USES BRESHENHAM 👇 USES BRESHENHAM 👇 USES BRESHENHAM 👇 USES BRESHENHAM 👇 USES BRESHENHAM 👇 USES BRESHENHAM
// USES BRESHENHAM 👇 USES BRESHENHAM 👇 USES BRESHENHAM 👇 USES BRESHENHAM 👇 USES BRESHENHAM 👇 USES BRESHENHAM

test "imageToys. render BoxPoly and filled cube" {

    // try mkdirIgnoreExists("filledCube");

    // const poly = mesh.BoxPoly.createAABB(.{0,0,0} , .{10,12,14});
    var cam = try PerspectiveCamera.init(
        .{ 100, 100, 100 },
        .{ 5, 5, 5 },
        401,
        301,
        null,
    );
    defer cam.deinit();

    var nx: u32 = 401;
    var ny: u32 = 301;

    cc.bres.init(&cam.screen[0], &ny, &nx);
    // print("\nvertices:\n{d:.2}",.{poly.vs});

    // const polyImage = try allocator.alloc(f32,10*12*14);
    var img = Img3D(f32){
        .img = try allocator.alloc(f32, 10 * 12 * 14),
        .nx = 14,
        .ny = 12,
        .nz = 10,
    };
    defer allocator.free(img.img);

    for (img.img) |*v, i| v.* = @intToFloat(f32, i);

    // print("minmax cam screen: {d}\n", .{im.minmax(f32,img.img)});

    perspectiveProjection2(img, &cam);
    try im.saveF32AsTGAGreyNormed(cam.screen, 301, 401, "filledCube.tga");

    // for (poly.es) |e,i| {
    //   // if (e[0]!=0) continue;
    //   const p0 = cam.world2pix(poly.vs[e[0]]);
    //   const p1 = cam.world2pix(poly.vs[e[1]]);
    //   // plotLine(f32,container,p0.x,p0.y,p1.x,p1.y,@intToFloat(f32,1));
    //   cc.bres.plotLine(@intCast(i32,p0.x) , @intCast(i32,p0.y) , @intCast(i32,p1.x) , @intCast(i32,p1.y));
    //   print("e={d}\n",.{e});
    //   print("p0={d:.2}\n",.{p0});
    //   print("p1={d:.2}\n",.{p1});
    //   name = try std.fmt.bufPrint(name, "filledCube/sides{:0>4}.tga", .{i});
    //   try im.saveF32AsTGAGreyNormed(allocator, cam.screen, 301, 401, name);
    // }
}

// Writes pixel values on curve to 1.0
pub fn randomBezierCurves(img: *Img2D(f32), ncurves: u16) void {

    // Set bresenham global state.
    var nx = img.nx;
    var ny = img.ny;
    cc.bres.init(&img.img[0], &ny, &nx);

    // Generate 100 random Bezier curve segments and draw them in the image.
    {
        var i: u16 = 0;
        while (i < ncurves) : (i += 1) {
            const x0 = @intCast(i32, rando.uintLessThan(u32, nx));
            const y0 = @intCast(i32, rando.uintLessThan(u32, ny));
            const x1 = @intCast(i32, rando.uintLessThan(u32, nx));
            const y1 = @intCast(i32, rando.uintLessThan(u32, ny));
            const x2 = @intCast(i32, rando.uintLessThan(u32, nx));
            const y2 = @intCast(i32, rando.uintLessThan(u32, ny));
            cc.bres.plotQuadBezier(x0, y0, x1, y1, x2, y2);
        }
    }
}

test "imageToys. color random Bezier curves with fillImg2D()" {
    var nx: u32 = 1900;
    var ny: u32 = 1024;

    // Initialize memory to 0. Set bresenham global state.
    var img = try allocator.alloc(f32, nx * ny);
    defer allocator.free(img);
    for (img) |*v| v.* = 0;
    cc.bres.init(&img[0], &ny, &nx);

    // Generate 100 random Bezier curve segments and draw them in the image.
    {
        var i: u16 = 0;
        while (i < 100) : (i += 1) {
            const x0 = @intCast(i32, rando.uintLessThan(u32, nx));
            const y0 = @intCast(i32, rando.uintLessThan(u32, ny));
            const x1 = @intCast(i32, rando.uintLessThan(u32, nx));
            const y1 = @intCast(i32, rando.uintLessThan(u32, ny));
            const x2 = @intCast(i32, rando.uintLessThan(u32, nx));
            const y2 = @intCast(i32, rando.uintLessThan(u32, ny));
            cc.bres.plotQuadBezier(x0, y0, x1, y1, x2, y2);
        }
    }

    // label the empty 0-valued spaces between bezier crossings
    var imgu8 = try allocator.alloc(u8, nx * ny);
    defer allocator.free(imgu8);
    for (imgu8) |*v, i| v.* = @floatToInt(u8, img[i]);
    const _res = try fillImg2D(imgu8, @intCast(u16, nx), @intCast(u16, ny));
    defer allocator.free(_res.img);

    const res = _res.img;
    const maxLabel = _res.maxLabel;
    print("MaxLabel = {}\n", .{maxLabel});

    // try countPixelValues(res);
    // try printPixelValueCounts(res);

    // std.sort.sort(u16 , res , {} , comptime std.sort.asc(u16)); // block sort
    // print("\n{d}\n",.{res[res.len-100..]});

    // Save the result as a greyscale image.
    for (imgu8) |*v, i| v.* = @intCast(u8, res[i] % 256);
    // try im.saveU8AsTGAGrey(allocator, imgu8, @intCast(u16, ny), @intCast(u16, nx), "fill_curve.tga");

    const rgbImg = try randomColorLabels(allocator, u8, imgu8);
    defer allocator.free(rgbImg);

    try im.saveU8AsTGA(rgbImg, @intCast(u16, ny), @intCast(u16, nx), "rgbFillImg.tga");
}

test "imageToys. save img of Random Bezier Curves" {
    print("\n", .{});

    // Initialize memory to 0.
    var img = Img2D(f32){
        .img = try allocator.alloc(f32, 1800 * 1200),
        .nx = 1800,
        .ny = 1200,
    };
    defer allocator.free(img.img);
    for (img.img) |*v| v.* = 0;

    // 10 Random Bezier Curves
    randomBezierCurves(&img, 10);

    // Convert to u8 and save as grescale image.
    // QUESTION: how can I make this memory `const` ? (or at least, const after the initialization?)
    var data = try allocator.alloc(u8, img.nx * img.ny);
    defer allocator.free(data);
    for (data) |*v, i| v.* = @floatToInt(u8, img.img[i] * 255);

    try im.saveU8AsTGAGrey(data, @intCast(u16, img.ny), @intCast(u16, img.nx), "randomBezierCurves.tga");
}
