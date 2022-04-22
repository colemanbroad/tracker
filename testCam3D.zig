// const std  = @import("std");
// const im   = @import("imageBase.zig");
// const geo  = @import("geometry.zig");
// const draw = @import("drawing.zig");

// const print = std.debug.print;
// const assert = std.debug.assert;
// const expect = std.testing.expect;
// var allocator = std.testing.allocator;
// const Allocator = std.mem.Allocator;
// var prng = std.rand.DefaultPrng.init(0);
// const random = prng.random();

// const Ray = geo.Ray;
// const Vec3 = geo.Vec3;
// const Mat3x3 = geo.Mat3x3;
// const Img3D = im.Img3D;
// const Img2D = im.Img2D;
// const BoxPoly = geo.BoxPoly;

// const abs = geo.abs;
// const sphereTrajectory = geo.sphereTrajectory;
// const normalize = geo.normalize;
// const drawLineInBounds = draw.drawLineInBounds;
// const cross = geo.cross;

usingnamespace @import("Cam3D.zig");

const draw = @import("drawing.zig");
const drawLineInBounds = draw.drawLineInBounds;

test "cam3D. render stars with perspectiveProjection2()" {
    // pub fn main() !void {

    // try mkdirIgnoreExists("renderStarsWPerspective");
    print("\n", .{});

    var img = try randomStars(allocator);
    defer allocator.free(img.img); // FIXME

    var nameStr = try std.fmt.allocPrint(allocator, "renderStarsWPerspective/img{:0>4}.tga", .{0});
    defer allocator.free(nameStr);

    const traj = sphereTrajectory();

    // {var i:u32=0; while(i<5):(i+=1){
    {
        var i: u32 = 0;
        while (i < traj.len) : (i += 1) {

            // const i_ = @intToFloat(f32,i);
            const camPt = traj[i] * Vec3{ 900, 900, 900 };
            // print("\n{d}",.{camPt});
            // const camPt = Vec3{400,50,50};
            var cam2 = try PerspectiveCamera.init(
                camPt,
                .{ 0, 0, 0 },
                401,
                301,
                null,
            );
            defer cam2.deinit();

            perspectiveProjection2(img, &cam2);

            nameStr = try std.fmt.bufPrint(nameStr, "renderStarsWPerspective/img{:0>4}.tga", .{i});
            print("{s}\n", .{nameStr});
            try im.saveF32AsTGAGreyNormed(cam2.screen, 301, 401, nameStr);
        }
    }

    // var nameStr = try std.fmt.allocPrint(allocator, "rotproj/projImagePointsPerspective{:0>4}.tga", .{0}); // filename
    // try im.saveU8AsTGAGrey(allocator, res, 100, 200, "projImagePoints.tga");
}

// img volume filled with stars. img shape is 50x100x200 .ZYX
// WARNING: must free() Img3D.img field
pub fn randomStars(allo: Allocator) !Img3D(f32) {
    var img = blo: {
        var data = try allo.alloc(f32, 50 * 100 * 200);
        for (data) |*v| v.* = 0;
        var img3d = Img3D(f32){
            .img = data,
            .nz = 50,
            .ny = 100,
            .nx = 200,
        };
        break :blo img3d;
    };

    // Generate 100 random 3D points. We include boundary conditions here! This is
    // shape dependent. Might be better to separate this out into a separate call to `clamp`.
    const nx = img.nx;
    const ny = img.ny;
    const nz = img.nz;
    const nxy = nx * ny;

    var i: u16 = 0;
    while (i < 100) : (i += 1) {
        const x0 = 1 + @intCast(u32, random.uintLessThan(u32, nx - 2));
        const y0 = 1 + @intCast(u32, random.uintLessThan(u32, ny - 2));
        const z0 = 1 + @intCast(u32, random.uintLessThan(u32, nz - 2));

        // Add markers as star
        img.img[z0 * nxy + y0 * nx + x0] = 1.0;
        img.img[z0 * nxy + y0 * nx + x0 - 1] = 1.0;
        img.img[z0 * nxy + y0 * nx + x0 + 1] = 1.0;
        img.img[z0 * nxy + (y0 - 1) * nx + x0] = 1.0;
        img.img[z0 * nxy + (y0 + 1) * nx + x0] = 1.0;
        img.img[(z0 - 1) * nxy + y0 * nx + x0] = 1.0;
        img.img[(z0 + 1) * nxy + y0 * nx + x0] = 1.0;
    }

    return img;
}

test "cam3D. test all PerspectiveCamera transformations" {
    //   try camTest();
    // }
    // pub fn camTest() !void {
    var cam = try PerspectiveCamera.init(
        .{ 100, 100, 100 },
        .{ 50, 50, 50 },
        401,
        301,
        null,
    );
    defer cam.deinit();

    print("\n", .{});

    try expect(@reduce(.And, cam.world2cam(cam.loc) == Vec3{ 0, 0, 0 }));
    print("cam.loc in cam coordinates : {d:.3} \n", .{cam.world2cam(cam.loc)});
    const p1 = cam.world2cam(cam.pointOfFocus);
    print("cam.pointOfFocus in cam coordinates : {d:.3} \n", .{p1});
    try expect(p1[1] == 0);
    try expect(p1[2] == 0);
    const p0 = cam.world2cam(.{ 0, 0, 0 }); // origin in cam coordinates
    print("origin in cam coordinates : {d:.3} \n", .{p0});
    print("origin back to world coordinates : {d:.3} \n", .{cam.cam2world(p0)});
    print("pointOfFocus : {d:.3} \n", .{cam.cam2world(cam.world2cam(cam.pointOfFocus))});

    print("\n\nFuzz Testing\n\n", .{});
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        const p2 = geo.randNormalVec3();
        // const z  = p2[0];
        // print("p2 = {d}\n",.{p2});

        const d0 = cam.cam2world(cam.world2cam(p2)) - p2;
        // print("c2w.w2c ... d0 = {d}\n",.{d0});
        try expect(abs(d0) < 1e-4);

        const d2 = cam.world2cam(cam.cam2world(p2)) - p2;
        // print("c2w.w2c ... d2 = {d}\n",.{d2});
        try expect(abs(d2) < 1e-4);

        const rpx = .{ .x = random.int(u8), .y = random.int(u8) };
        const pt2 = cam.pix2world(rpx);
        const pt3 = (pt2 - cam.loc) * Vec3{ 1, 1, 1 } + cam.loc;
        const px2 = cam.world2pix(pt3);
        // print("rpx : {d}\n", .{rpx});
        print("px2 : {d}\n", .{px2});

        // const px0 = cam.world2pix(p2);
        // print("px0 = {d}\n",.{px0});
        // const pt3 = cam.pix2world(px0);
        // print("pt3 = {d}\n",.{pt3});

        // const r1 = Ray{.pt0=cam.loc , .pt1=pt3};
        // const d1 = closestApproachRayPt(r1,p2);
        // print("w2p.p2w ... d1 = {d}\n",.{d1});
        // try expect(abs(d1-p2) < 1e-4);

    }
}
