const std = @import("std");
const im = @import("image_base.zig");
const geo = @import("geometry.zig");
const mesh = @import("mesh.zig");

const print = std.debug.print;
const assert = std.debug.assert;
const expect = std.testing.expect;

const random = prng.random();
var allocator = std.testing.allocator;
var prng = std.rand.DefaultPrng.init(0);

const Allocator = std.mem.Allocator;
const Ray = geo.Ray;
const Vec3 = geo.Vec3;
const Mat3x3 = geo.Mat3x3;
const Img3D = im.Img3D;
const Img2D = im.Img2D;
const BoxPoly = mesh.BoxPoly;
const pi = 3.1415926;

// const Mesh3D = mesh.Mesh3D;

// const sphereTrajectory = geo.sphereTrajectory;
const l2norm = geo.l2norm;
const normalize = geo.normalize;
const cross = geo.cross;

pub fn perspectiveProjection2(image: Img3D(f32), cam: *PerspectiveCamera) void {

    // Camera and view parameters
    // const camPt = normalize(camPt_);
    const box = Vec3{ @intToFloat(f32, image.nz), @intToFloat(f32, image.ny), @intToFloat(f32, image.nx) };
    const boxMidpoint = box / @splat(3, @as(f32, 2));
    cam.refocus(cam.loc, boxMidpoint);

    var projmax: f32 = 0;

    // Loop over the projection screen. For each pixel in the screen we cast a ray out along `z` in the screen coordinates,
    // and take the max value along that ray.
    var iy: u32 = 0;
    while (iy < cam.nyPixels) : (iy += 1) {
        const y2 = iy * cam.nxPixels;
        var ix: u32 = 0;

        while (ix < cam.nxPixels) : (ix += 1) {

            // const r0 = cam.transPixel2World(ix,iy,1);  // returns Ray from camera pixel in World Coordinates
            const v0 = cam.pix2world(.{ .x = ix, .y = iy });
            // Test for intersection points of Ray with image volume.
            // Skip this ray if we don't intersect box.
            var intersection = geo.intersectRayAABB(Ray{ .pt0 = cam.loc, .pt1 = v0 }, Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = box });
            // var intersection = intersectRayAABB( r0 , Ray{.pt0=.{0,0,0} , .pt1=box} ); // returns Ray with 2 intersection points
            const v1 = normalize(v0 - cam.loc);

            // skip this Ray if it doesn't intersect with box
            if (intersection.pt0 == null and intersection.pt1 == null) continue;
            if (intersection.pt0 == null) intersection.pt0 = cam.loc; // equal to r0.pt0;
            if (intersection.pt1 == null) intersection.pt1 = cam.loc; // equal to r0.pt0;
            // Now that both points are well defined we can cast rays from inside the volume!

            // Take maximum value along the intersecting line segment.
            const intersectionLength = l2norm(intersection.pt1.? - intersection.pt0.?);
            var iz_f: f32 = 3;
            while (iz_f < intersectionLength - 1) : (iz_f += 1) {

                // NOTE: We start at intersection.pt0 always, because we define it to be nearest to starting point.
                // v0 is unit vec pointing away from camera
                const v4 = intersection.pt0.? + v1 * @splat(3, iz_f);

                // print("p0 = {d}\n",.{intersection.pt0.?});
                // print("v1 = {d}\n",.{v1});
                // print("v4 = {d}\n",.{v4});

                // if @reduce(.Or, v4<0 or v4>box-1) continue; // TODO: The compiler will probably allow this one day.
                // if (@reduce(.Or, v4<@splat(3,@as(f32,0))) or @reduce(.Or,v4>box - @splat(3,@as(f32,1)))) continue; // TODO: The compiler will probably allow this one day.

                if (v4[0] < 0 or v4[0] > box[0] - 1) continue; //{break :blk null;}
                if (v4[1] < 0 or v4[1] > box[1] - 1) continue; //{break :blk null;}
                if (v4[2] < 0 or v4[2] > box[2] - 1) continue; //{break :blk null;}

                const val = interp3DLinear(image, v4[0], v4[1], v4[2]);

                if (cam.screen[y2 + ix] < val) {
                    cam.screen[y2 + ix] = val;
                    if (val > projmax) projmax = val;
                }
            }
        }
    }

    // Add bounding box consisting of 8 vertices connected by 12 lines separating 6 faces
    // the 12 lines come from the following. We have a low box corner [0,0,0] and a high corner [nx,ny,nz].
    // If we start with choosing the low value in each of the x,y,z dimensions we can get the next three lines
    // by choosing the high value in exactly one of the three dims, i.e. [0,0,0] is connected to [1,0,0],[0,1,0],[0,0,1] which each sum to 1
    // and [1,0,0] is connected to [0,0,0] and [1,1,0],[1,0,1] which sum to 2.
    // The pattern we are describing is a [Hasse Diagram](https://mathworld.wolfram.com/HasseDiagram.html)

    // var nx:u32=cam.nxPixels;
    // var ny:u32=cam.nyPixels;
    // cc.bres.init(&cam.screen[0],&ny,&nx);
    const container = Img2D(f32){ .img = cam.screen, .nx = cam.nxPixels, .ny = cam.nyPixels };
    const poly = BoxPoly.createAABB(.{ 0, 0, 0 }, box);
    for (poly.es) |e| {
        const p0 = cam.world2pix(poly.vs[e[0]]);
        const p1 = cam.world2pix(poly.vs[e[1]]);
        // plotLine(f32,container,p0.x,p0.y,p1.x,p1.y,1.0);
        // cc.bres.plotLine(@intCast(i32,p0.x) , @intCast(i32,p0.y) , @intCast(i32,p1.x) , @intCast(i32,p1.y));
        const x0 = @intCast(i32, p0.x);
        const y0 = @intCast(i32, p0.y);
        const x1 = @intCast(i32, p1.x);
        const y1 = @intCast(i32, p1.y);
        im.drawLineInBounds(f32, container, x0, y0, x1, y1, projmax);
    }

    // DONE
}

// const test_home = @import("tester.zig").test_path;

test {
    std.testing.refAllDecls(@This());
}

const AxisOrder = enum { XYZ, ZYX };

// Construct an orthogonal rotation matrix which aligns z->camera and y->z
// `camPt_` uses 👇 normalized coordinates and points toward the origin [0,0,0].
fn cameraRotation(camPt_: Vec3, axisOrder: AxisOrder) Mat3x3 {

    // standardize on XYZ axis order
    const camPt = switch (axisOrder) {
        .XYZ => camPt_,
        .ZYX => Vec3{ camPt_[2], camPt_[1], camPt_[0] },
    };

    const x1 = Vec3{ 1, 0, 0 };
    // const y1 = Vec3{0,1,0};
    const z1 = Vec3{ 0, 0, 1 };
    const z2 = normalize(camPt);
    const x2 = if (@reduce(.And, z1 == z2)) x1 else normalize(cross(z1, z2)); // protect against z1==z2.
    const y2 = normalize(cross(z2, x2));

    const rotM = geo.matFromVecs(x2, y2, z2);

    // return in specified axis order
    switch (axisOrder) {
        .XYZ => return rotM,
        .ZYX => return Mat3x3{
            rotM[8],
            rotM[7],
            rotM[6],
            rotM[5],
            rotM[4],
            rotM[3],
            rotM[2],
            rotM[1],
            rotM[0],
        },
    }
}

// pass in pointer-to-array or slice
fn reverse_inplace(array: anytype) void {
    var temp = array[0];
    const n = array.len;
    var i: usize = 0;

    while (i < @divFloor(n, 2)) : (i += 1) {
        temp = array[i];
        array[i] = array[n - i];
        array[n - i] = temp;
    }
}

test "test cameraRotation()" {

    // begin with XYZ coords, then swap to ZYX
    const x1 = Vec3{ 1, 0, 0 };
    const y1 = Vec3{ 0, 1, 0 };
    const z1 = Vec3{ 0, 0, 1 };

    const camPt = Vec3{ -1, -1, -1 }; // checks
    const z2 = normalize(camPt);
    const x2 = normalize(geo.cross(z1, z2));
    const y2 = normalize(geo.cross(z2, x2));

    const rotM = geo.matFromVecs(x2, y2, z2);

    print("\n", .{});
    print("Rotated x1: {d} \n", .{geo.matVecMul(rotM, x1)});
    print("Rotated y1: {d} \n", .{geo.matVecMul(rotM, y1)});
    print("Rotated z1: {d} \n", .{geo.matVecMul(rotM, z1)});

    try expect(x2[2] == 0);
    try expect(l2norm(cross(x2, y2) - z2) < 1e-6);
}

const SCREENX: u16 = 2880;
const SCREENY: u16 = 1800;

// // NOTE: By convention, the camera faces the -z direction, which allows y=UP, x=RIGHT in a right handed coordinate system.
pub const PerspectiveCamera = struct {
    loc: Vec3, // location of camera in world coordinates
    pointOfFocus: Vec3, // point of focus in world coordinates (no simulated focal plane, so any ray is equivalent for now.)
    nxPixels: u32, // Assume square pixels. nx/ny defines the aspect ratio (aperture)
    nyPixels: u32, // Assume square pixels. nx/ny defines the aspect ratio (aperture)
    apertureX: f32, // Field of view in the horizontal (x) direction. Units s.t. fov=1 produces 62° view angle. nx/ny_pixels determines FOV in y (vertical direction).
    apertureY: f32,
    axes: Mat3x3, // orthonormal axes of camera coordinate system (in world coordinates)
    axesInv: Mat3x3,
    screen: []f32, // where picture data is recorded
    // screenFace : [4]Vec3, // four points which define aperture polygon (in world coordinates)

    pub fn init(
        loc: Vec3,
        pointOfFocus: Vec3,
        nxPixels: u32,
        nyPixels: u32,
        _apertureX: ?f32,
    ) !@This() {
        var apertureX = if (_apertureX) |fx| fx else 0.2;
        var apertureY = apertureX * @intToFloat(f32, nyPixels) / @intToFloat(f32, nxPixels);
        var screen = try allocator.alloc(f32, nxPixels * nyPixels);
        var axes = cameraRotation(loc - pointOfFocus, .ZYX);
        for (axes) |v| assert(v != std.math.nan_f32);
        var axesInv = geo.invert3x3(axes);
        for (axesInv) |v| assert(v != std.math.nan_f32);

        var this = @This(){
            .loc = loc,
            .pointOfFocus = pointOfFocus,
            .nxPixels = nxPixels,
            .nyPixels = nyPixels,
            .apertureX = apertureX,
            .apertureY = apertureY,
            .axes = axes,
            .axesInv = axesInv,
            .screen = screen,
            // .screenFace=undefined,
        };

        // // update screenFace
        // var sf0 = this.pix2world(0,0);
        // var sf1 = this.pix2world(nxPixels,0);
        // var sf2 = this.pix2world(nxPixels,nyPixels);
        // var sf3 = this.pix2world(0,nyPixels);
        // this.screenFace = .{sf0,sf1,sf2,sf2};

        return this;
    }

    pub fn deinit(this: @This()) void {
        allocator.free(this.screen);
    }

    // world2cam(cam.loc) = {0,0,0}
    // world2cam(cam.pointOfFocus) = {dist,0,0}
    pub fn world2cam(this: @This(), v0: Vec3) Vec3 {
        const v1 = v0 - this.loc;
        const v2 = geo.matVecMul(this.axesInv, v1);
        return v2;
    }

    // cam2world({0,0,0}) = cam.loc
    // cam2world(worldOrigin) = {0,0,0}
    // cam2world(focalPoint) = cam.pointOfFocus
    pub fn cam2world(this: @This(), v0: Vec3) Vec3 {
        // const p0 = Vec3{-l2norm(this.loc), 0 , 0}; // location of world origin in camera coordinates
        const v1 = geo.matVecMul(this.axes, v0);
        const v2 = v1 + this.loc;
        return v2;
        // const v1 = v0 + this.loc; // translate origin to [0,0,0]
        // const v2 = matVecMul(this.axes, v1); // rotate
    }

    const Px = struct { x: i64, y: i64 };

    // returns [x,y] pixel coordinates
    pub fn world2pix(this: @This(), v0: Vec3) Px {
        const v1 = this.world2cam(v0);
        const v2 = v1 / @splat(3, -v1[0]); // divide by -Z to normalize to -1 (homogeneous coords)
        const ny = @intToFloat(f32, this.nyPixels);
        const nx = @intToFloat(f32, this.nxPixels);
        const v3 = (v2 - Vec3{ 0, -this.apertureY / 2, -this.apertureX / 2 }) / Vec3{ 1, this.apertureY, this.apertureX } * Vec3{ 1, ny, nx };
        // const v3 = (v2 + Vec3{0,this.apertureY/2,this.apertureX/2}) * Vec3{1,ny,nx} / Vec3{1,this.apertureY,this.apertureX};
        const v4 = @floor(v3);
        const y = @floatToInt(i64, v4[1]);
        const x = @floatToInt(i64, v4[2]);
        return .{ .x = x, .y = y };
    }

    // a pixel (input) collects light from all points along a Ray (return)
    // NOTE: px are allowed to be i64 (we can refer to px outside the image boundaries)
    pub fn pix2world(this: @This(), px: Px) Vec3 {
        const _x = (@intToFloat(f32, px.x) / @intToFloat(f32, this.nxPixels - 1) - 0.5) * this.apertureX; // map pixel values onto [-0.5,0.5] inclusive
        const _y = (@intToFloat(f32, px.y) / @intToFloat(f32, this.nyPixels - 1) - 0.5) * this.apertureY; // map pixel values onto [-0.5,0.5] inclusive
        const v0 = Vec3{ -1, _y, _x };
        const v1 = this.cam2world(v0);
        // return Ray{.pt0=this.loc, .pt1=v1};
        return v1;
    }

    pub fn refocus(this: *@This(), loc: Vec3, pointOfFocus: Vec3) void {
        this.axes = cameraRotation(loc - pointOfFocus, .ZYX);
        this.axesInv = geo.invert3x3(this.axes); // TODO: this should just be a transpose
        this.loc = loc;
        this.pointOfFocus = pointOfFocus;
    }
};

// Classical lerp
inline fn interp3DLinear(image3D: Img3D(f32), z: f32, y: f32, x: f32) f32 {
    const nxy = image3D.nx * image3D.ny;
    const nx = image3D.nx;
    const img = image3D.img;

    // Get indices of six neighbouring hyperplanes
    const z0 = @floatToInt(u32, @floor(z));
    const z1 = @floatToInt(u32, @ceil(z));
    const y0 = @floatToInt(u32, @floor(y));
    const y1 = @floatToInt(u32, @ceil(y));
    const x0 = @floatToInt(u32, @floor(x));
    const x1 = @floatToInt(u32, @ceil(x));

    // delta vec
    const dz = z - @floor(z);
    const dy = y - @floor(y);
    const dx = x - @floor(x);

    // Get values of eight neighbouring pixels
    const f000 = img[z0 * nxy + y0 * nx + x0];
    const f001 = img[z0 * nxy + y0 * nx + x1];
    const f010 = img[z0 * nxy + y1 * nx + x0];
    const f011 = img[z0 * nxy + y1 * nx + x1];
    const f100 = img[z1 * nxy + y0 * nx + x0];
    const f101 = img[z1 * nxy + y0 * nx + x1];
    const f110 = img[z1 * nxy + y1 * nx + x0];
    const f111 = img[z1 * nxy + y1 * nx + x1];

    // ┌───┐              ┌───┐
    // │x10│              │x11│
    // └───┘▲───────────▶▲└───┘
    //      │            │
    //      │            │
    //      │            │      example edge     ┌──────┐
    //      │            │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│x01_11│
    //      │            │                       └──────┘
    //      │            │
    //      │───────────▶│
    // ┌───┐            ┌───┐
    // │x00│            │x01│
    // └───┘            └───┘

    // Linear interp along Z
    const f00 = dz * f100 + (1 - dz) * f000;
    const f01 = dz * f101 + (1 - dz) * f001;
    const f10 = dz * f110 + (1 - dz) * f010;
    const f11 = dz * f111 + (1 - dz) * f011;

    // Linearly interp along Y
    const f01_11 = (1 - dy) * f01 + dy * f11;
    const f00_10 = (1 - dy) * f00 + dy * f10;

    // Linear interp along X
    const result = (1 - dx) * f00_10 + dx * f01_11;

    // const f00_01 = (1-dx)*f00 + dx*f01;
    // const f10_11 = (1-dx)*f10 + dx*f11;
    // Linearly interp along Y
    // const f_x1 = (1-dy)*f00_01 + dy*f10_11;
    // const f_x2 = (1-dy)*f00_01 + dy*f10_11;
    // const f_y_final = (1-dx)*f00_10 + dx*f01_11;
    // const result = (f_x_final + f_y_final) / 2;

    return result;
}

test "cam3D. test trilinear interp" {
    print("\n", .{});

    // Build a 2x2x2 image with the pixel values 1..8
    const img = blo: {
        var _a = [_]f32{
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        };
        // var _a  = [_]f32{5,5,5,5,5,5,5,5,};
        var _b = Img3D(f32){
            .img = &_a,
            .nz = 2,
            .ny = 2,
            .nx = 2,
        };
        break :blo _b;
    };

    // Assert that the middle interpolated value is the average across all 8 pixels.
    var r1 = interp3DLinear(img, 0.5, 0.5, 0.5);
    try std.testing.expect((r1 > 1) and (r1 < 8));
}

// spiral walk around the unit sphere
pub fn sphereTrajectory() [100]Vec3 {
    var phis: [100]f32 = undefined;
    // for (phis) |*v,i| v.* = ((@intToFloat(f32,i)+1)/105) * pi;
    for (&phis, 0..) |*v, i| v.* = ((@intToFloat(f32, i)) / 99) * pi;
    var thetas: [100]f32 = undefined;
    // for (&thetas) |*v,i| v.* = ((@intToFloat(f32,i)+1)/105) * 2*pi;
    for (&thetas, 0..) |*v, i| v.* = ((@intToFloat(f32, i)) / 99) * 2 * pi;
    var pts: [100]Vec3 = undefined;
    for (&pts, 0..) |*v, i| v.* = Vec3{ @cos(phis[i]), @sin(thetas[i]) * @sin(phis[i]), @cos(thetas[i]) * @sin(phis[i]) }; // ZYX coords
    return pts;
}

const test_home = "/Users/broaddus/work/isbi/zig-tracker/test-artifacts/render/";

test "cam3D. render stars with perspectiveProjection2()" {
    // pub fn main() !void {

    if (true) return error.SkipZigTest;

    // try mkdirIgnoreExists("renderStarsWPerspective");
    print("\n", .{});

    var img = try randomStars(allocator);
    defer allocator.free(img.img); // FIXME

    var nameStr = try std.fmt.allocPrint(allocator, test_home ++ "renderStarsWPerspective/img{:0>4}.tga", .{0});
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

            nameStr = try std.fmt.bufPrint(nameStr, test_home ++ "renderStarsWPerspective/img{:0>4}.tga", .{i});
            print("{s}\n", .{nameStr});

            // @breakpoint();

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
        try expect(l2norm(d0) < 1e-4);

        const d2 = cam.world2cam(cam.cam2world(p2)) - p2;
        // print("c2w.w2c ... d2 = {d}\n",.{d2});
        try expect(l2norm(d2) < 1e-4);

        const rpx = .{ .x = random.int(u8), .y = random.int(u8) };
        const pt2 = cam.pix2world(rpx);
        const pt3 = (pt2 - cam.loc) * Vec3{ 1, 1, 1 } + cam.loc;
        const px2 = cam.world2pix(pt3);
        // print("rpx : {d}\n", .{rpx});
        print("px2 : {any}\n", .{px2});

        // const px0 = cam.world2pix(p2);
        // print("px0 = {d}\n",.{px0});
        // const pt3 = cam.pix2world(px0);
        // print("pt3 = {d}\n",.{pt3});

        // const r1 = Ray{.pt0=cam.loc , .pt1=pt3};
        // const d1 = closestApproachRayPt(r1,p2);
        // print("w2p.p2w ... d1 = {d}\n",.{d1});
        // try expect(l2norm(d1-p2) < 1e-4);

    }
}
