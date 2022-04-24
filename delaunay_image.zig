const std = @import("std");
const im = @import("imageBase.zig");
// const cc = @import("c.zig");

const geo = @import("geometry.zig");
const draw = @import("drawing_basic.zig");

var allocator = std.testing.allocator;
var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

const print = std.debug.print;
const assert = std.debug.assert;
const expect = std.testing.expect;
const clamp = std.math.clamp;

const Img2D = im.Img2D;
const Img3D = im.Img3D;

const process = std.process;
const Vec2 = geo.Vec2;

const Vector = std.meta.Vector;
const del = @import("delaunay.zig");

// const home = @import("tester.zig").test_path ++ "/delaunay_image/";
const home = "/Users/broaddus/Desktop/work/zig-tracker/test-artifacts/delaunay_image/";

test {
    std.testing.refAllDecls(@This());
}

// pub fn main() !void {
test "delaunay_image. triangulation in 2D" {
    badtrianglesGlobal = try std.ArrayList(usize).initCapacity(allocator, 100);
    defer badtrianglesGlobal.deinit();

    // var arg_it = process.args();
    // _ = arg_it.skip(); // skip exe name
    // const npts_str = try arg_it.next(allocator) orelse "300";
    // const nparticles = try std.fmt.parseUnsigned(usize, npts_str, 10);

    // var arg_it = try process.argsWithAllocator(allocator);
    // _ = arg_it.skip(); // skip exe name
    // const npts_str = arg_it.next() orelse "100";
    // const nparticles = try std.fmt.parseUnsigned(usize, npts_str, 10);

    const nparticles = if (@import("builtin").is_test) 1000 else blk: {
        var arg_it = try process.argsWithAllocator(allocator);
        _ = arg_it.skip(); // skip exe name
        const npts_str = arg_it.next() orelse "100";
        break :blk try std.fmt.parseUnsigned(usize, npts_str, 10);
    };

    // blank picture
    var pic = try Img2D([4]u8).init(1200, 1200);
    defer pic.deinit();
    for (pic.img) |*v| v.* = .{ 0, 0, 0, 255 };

    // create random vertices
    const verts = try allocator.alloc(Vec2, nparticles); // 10k requires too much memory for triangles. The scaling is nonlinear.
    defer allocator.free(verts); // changes size when we call delaunay2d() ????
    for (verts) |*v| {
        const x = random.float(f32) * 1200;
        const y = random.float(f32) * 1200;
        v.* = .{ x, y };
    }

    // perform delaunay triangulation
    const triangles = try del.delaunay2d(allocator, verts);
    defer allocator.free(triangles);

    // draw the triangulation
    for ([_]u8{ 0, 1 }) |v| {
        for (triangles) |tri| {
            const x0 = @floatToInt(u31, verts[tri[0]][0]);
            const y0 = @floatToInt(u31, verts[tri[0]][1]);
            const x1 = @floatToInt(u31, verts[tri[1]][0]);
            const y1 = @floatToInt(u31, verts[tri[1]][1]);
            const x2 = @floatToInt(u31, verts[tri[2]][0]);
            const y2 = @floatToInt(u31, verts[tri[2]][1]);

            if (v == 0) {
                draw.drawLine([4]u8, pic, x0, y0, x1, y1, .{ 255, 255, 255, 255 });
                draw.drawLine([4]u8, pic, x1, y1, x2, y2, .{ 255, 255, 255, 255 });
                draw.drawLine([4]u8, pic, x2, y2, x0, y0, .{ 255, 255, 255, 255 });
            }

            if (v == 1) {
                draw.drawCircle([4]u8, pic, x0, y0, 2, .{ 255, 50, 50, 255 });
                draw.drawCircle([4]u8, pic, x1, y1, 2, .{ 255, 50, 50, 255 });
                draw.drawCircle([4]u8, pic, x2, y2, 2, .{ 255, 50, 50, 255 });
            }
        }
    }
    // @breakpoint();

    // checkDelaunay(verts,triangles);

    try im.saveRGBA(pic, home ++ "delaunay-start.tga");
}

pub fn contains(comptime T: type, arr: []const T, val: T) bool {
    for (arr) |v| if (val == v) return true;
    return false;
}

var badtrianglesGlobal: std.ArrayList(usize) = undefined;

fn checkDelaunay(idx_endpt: usize, pts: []Vec2, triangles: []?[3]u32) !bool {
    var failure = false;
    for (pts[0..idx_endpt]) |pt, pt_idx| {
        for (triangles) |_tri, tri_idx| {
            if (!contains(usize, badtrianglesGlobal.items, tri_idx)) {
                if (_tri) |tri| { // triangle isn't null
                    if (!contains(u32, &tri, @intCast(u32, pt_idx))) { // triangle doesn't contain pt as vertex
                        const tripts = [3]Vec2{ pts[tri[0]], pts[tri[1]], pts[tri[2]] };
                        if (geo.pointInTriangleCircumcircle2d(pt, tripts)) { // pt is inside triangle circumcircle
                            print("pt_idx={d} , tri={d} \n", .{ pt_idx, tri });
                            failure = true;
                            try showdelaunaystate(pts, triangles, pt_idx);
                            try showdelaunaystate(pts, triangles, tri[0]);
                            try showdelaunaystate(pts, triangles, tri[1]);
                            try showdelaunaystate(pts, triangles, tri[2]);
                            badtrianglesGlobal.appendAssumeCapacity(tri_idx);
                        }
                    }
                }
            }
        }
    }
    return failure;
}

pub fn drawValidTriangles(pic: anytype, verts: []Vec2, triangles: [][3]u32, idx_pt: usize) void {
    // for (verts) |v| {
    //     const x0 = std.math.clamp(@floatToInt(u31, v[0]), 0, @intCast(u31, pic.nx) - 1);
    //     const y0 = std.math.clamp(@floatToInt(u31, v[1]), 0, @intCast(u31, pic.ny) - 1);
    //     // print("verts {d}:{d},{d} ... {d}\n", .{i,x0,y0,v});
    // }

    for (triangles) |tri| {
        const x0 = @floatToInt(u31, std.math.clamp(verts[tri[0]][0], 0, @intToFloat(f32, pic.nx) - 1));
        const y0 = @floatToInt(u31, std.math.clamp(verts[tri[0]][1], 0, @intToFloat(f32, pic.ny) - 1));
        const x1 = @floatToInt(u31, std.math.clamp(verts[tri[1]][0], 0, @intToFloat(f32, pic.nx) - 1));
        const y1 = @floatToInt(u31, std.math.clamp(verts[tri[1]][1], 0, @intToFloat(f32, pic.ny) - 1));
        const x2 = @floatToInt(u31, std.math.clamp(verts[tri[2]][0], 0, @intToFloat(f32, pic.nx) - 1));
        const y2 = @floatToInt(u31, std.math.clamp(verts[tri[2]][1], 0, @intToFloat(f32, pic.ny) - 1));
        // print("tri {d}:{d}\n", .{i,tri});

        // TODO: remove `clamp` and use drawLineInBounds()
        draw.drawLine([4]u8, pic, x0, y0, x1, y1, .{ 255, 255, 255, 255 });
        draw.drawLine([4]u8, pic, x1, y1, x2, y2, .{ 255, 255, 255, 255 });
        draw.drawLine([4]u8, pic, x2, y2, x0, y0, .{ 255, 255, 255, 255 });

        // new point
        var color: [4]u8 = undefined;
        color = if (tri[0] == idx_pt) .{ 25, 50, 250, 255 } else .{ 255, 50, 50, 255 };
        draw.drawCircle([4]u8, pic, x0, y0, 2, color);
        color = if (tri[1] == idx_pt) .{ 25, 50, 250, 255 } else .{ 255, 50, 50, 255 };
        draw.drawCircle([4]u8, pic, x1, y1, 2, color);
        color = if (tri[2] == idx_pt) .{ 25, 50, 250, 255 } else .{ 255, 50, 50, 255 };
        draw.drawCircle([4]u8, pic, x2, y2, 2, color);

        const triangleVertices = .{ verts[tri[0]], verts[tri[1]], verts[tri[2]] };
        const bigcircle = geo.getCircumcircle2d(triangleVertices);
        const ux = @floatToInt(i32, bigcircle.pt[0]);
        const uy = @floatToInt(i32, bigcircle.pt[1]);
        const r = @floatToInt(i32, @sqrt(bigcircle.r2));
        // drawCircle([4]u8,pic,ux,uy,r,.{100,100,0,255});

        // var colorcircle:[4]u8 = undefined;
        if (tri[0] == idx_pt or tri[1] == idx_pt or tri[2] == idx_pt)
            draw.drawCircleOutline(pic, ux, uy, r, .{ 25, 50, 250, 255 });
    }
}

var globalidx: u32 = 0;

pub fn showdelaunaystate(verts: []Vec2, triangles: []?[3]u32, idx_pt: usize) !void {
    const v1 = verts.len - 3;
    // remove null triangles
    var idx_valid: u32 = 0;
    var validtriangles = try allocator.alloc([3]u32, triangles.len);
    for (triangles) |tri| {
        if (tri) |vt| { // valid_triangle
            if (vt[0] >= v1 or vt[1] >= v1 or vt[2] >= v1) continue; // remove starting points
            validtriangles[idx_valid] = vt;
            idx_valid += 1;
        }
    }
    validtriangles = validtriangles[0..idx_valid];

    // wipe old pic, draw graph, and save img
    var pic = try Img2D([4]u8).init(1200, 1200);
    for (pic.img) |*v| v.* = .{ 0, 0, 0, 255 };

    drawValidTriangles(pic, verts, validtriangles, idx_pt);
    var name = try allocator.alloc(u8, 40);
    defer allocator.free(name);
    name = try std.fmt.bufPrint(name, "delaunay/img{:0>4}.tga", .{globalidx});
    try im.saveRGBA(pic, name);
    globalidx += 1;
}
