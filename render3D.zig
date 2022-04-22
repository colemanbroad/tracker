const std = @import("std");
const im = @import("imageBase.zig");
const geo = @import("geometry.zig");

const print = std.debug.print;
const assert = std.debug.assert;
const expect = std.testing.expect;
var allocator = std.testing.allocator;
const Allocator = std.mem.Allocator;
var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

const Ray = geo.Ray;
const Vec3 = geo.Vec3;
const Mat3x3 = geo.Mat3x3;
const Img3D = im.Img3D;
const Img2D = im.Img2D;
const BoxPoly = geo.BoxPoly;

const abs = geo.abs;
const sphereTrajectory = geo.sphereTrajectory;
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
            const intersectionLength = abs(intersection.pt1.? - intersection.pt0.?);
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
        draw.drawLineInBounds(f32, container, x0, y0, x1, y1, projmax);
    }

    // DONE
}
