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

const test_home = "/Users/broaddus/Desktop/work/zig-tracker/test-artifacts/drawing/";

test {
    std.testing.refAllDecls(@This());
}

pub inline fn inbounds(img: anytype, px: anytype) bool {
    if (0 <= px[0] and px[0] < img.nx and 0 <= px[1] and px[1] < img.ny) return true else return false;
}

// render mesh on top of pic. all white. fill in lines and faces. pure zig.
pub fn drawMesh2D(mesh: Mesh2D, pic: Img2D([4]u8)) !void {}

// render a 3D surface from a spiral trajectory
pub fn drawMesh3DMovie2(surf: Mesh, name: []const u8) !void {
    const pic = try Img2D(f32).init(800, 600);
    defer pic.deinit();

    const spheretraj = sphereTrajectory();
    var namestr = try allocator.alloc(u8, 100);
    defer allocator.free(namestr);

    const verts2d = try allocator.alloc([2]f32, surf.vs.len);
    defer allocator.free(verts2d);

    for (spheretraj) |pt, i| {
        const verts = try rotate2cam(allocator, surf.vs, pt); // TODO: allow preallocation
        defer allocator.free(verts);

        const indices = try faceZOrder(verts, surf.fs.?);
        defer allocator.free(indices);

        for (verts) |v, j| {
            const vz = v / Vec3{ v[0], v[0], v[0] }; // homogeneous coords
            var x = (vz[2] - 0) * 7000 + @intToFloat(f32, pic.nx) / 2;
            var y = (vz[1] - 0) * 7000 + @intToFloat(f32, pic.ny) / 2;
            // print("x,y={d},{d}\n", .{x,y});

            verts2d[j] = .{ x, y }; // ZYX -> XY

        }

        // fitboxiso(verts2d, pic.nx , pic.ny); // centers object in view

        // try drawFacesWithEdges(pic,verts2d,surf.fs.?); // TODO: allow preallocation
        try drawFaces(pic, verts2d, surf.fs.?, indices, false); // TODO: allow preallocation
        // drawEdges(pic,verts2d,surf.es);
        // try blurfilter(pic);
        // try minfilter(pic);

        namestr = try std.fmt.bufPrint(namestr, "{s}{:0>4}.tga", .{ name, i });
        try im.saveF32Img2D(pic, namestr); // TODO: allow preallocation (of RGBA img)
        for (pic.img) |*v| v.* = 0;
    }
}

// A slice of points in [x1 y1 x2 y2 x3 y3 ...] format.
// x,y must be in range [0,1].
// optionally draw lines connecting points in sequence
pub fn drawPoints2D(comptime T: type, points2D: []T, picname: []const u8, lines: bool) !void {

    // prepare canvas
    var nx: u32 = 2880;
    var ny: u32 = 1800;

    var canvas = try allocator.alloc(f32, nx * ny);
    for (canvas) |*v| v.* = 0; //if (i%4==3) 255 else 0;

    cc.bres.init(&canvas[0], &ny, &nx);

    // Normalize points to [0,nx],[0,ny]
    assert(points2D.len % 2 == 0);
    var xpts = try allocator.alloc(T, points2D.len / 2);
    var ypts = try allocator.alloc(T, points2D.len / 2);
    for (xpts) |_, i| {
        xpts[i] = points2D[2 * i];
        ypts[i] = points2D[2 * i + 1];
    }
    const xmima = im.minmax(T, xpts);
    const ymima = im.minmax(T, ypts);
    assert(xmima[1] > xmima[0]);
    assert(ymima[1] > ymima[0]);
    for (xpts) |_, i| {
        xpts[i] = (xpts[i] - xmima[0]) / (xmima[1] - xmima[0]) * @intToFloat(T, nx - 10) + 5;
        ypts[i] = (ypts[i] - ymima[0]) / (ymima[1] - ymima[0]) * @intToFloat(T, ny - 10) + 5;
    }

    var oldpt = [2]T{ xpts[0], ypts[0] }; // needed if lines==True

    {
        var i: u32 = 0;
        while (i < xpts.len) : (i += 1) {

            // snap to pixel grid
            const x_u16 = @floatToInt(u16, xpts[i]);
            const y_u16 = @floatToInt(u16, ypts[i]);
            const x = xpts[i];
            const y = ypts[i];

            // const idx = @floatToInt(usize, y*nx + x);
            const idx = y_u16 * nx + x_u16; //@floatToInt(usize, y*@intToFloat(f32,nx) + x);
            cc.bres.setPixel(@floatToInt(i32, x), @floatToInt(i32, y));
            cc.bres.setPixel(@floatToInt(i32, x), @floatToInt(i32, y));

            // white points. single pixel.
            canvas[idx + 0] = 1;
            // canvas[idx+1] = 1;
            // canvas[idx+2] = 1;
            // canvas[idx+3] = 1;

            // canvas[(idx+1)+0] = 1;
            // canvas[(idx+1)+1] = 1;
            // canvas[(idx+1)+2] = 1;
            // canvas[(idx+1)+3] = 1;

            // canvas[(idx-1)+0] = 1;
            // canvas[(idx-1)+1] = 1;
            // canvas[(idx-1)+2] = 1;
            // canvas[(idx-1)+3] = 1;

            // canvas[(idx+nx)+0] = 1;
            // canvas[(idx+nx)+1] = 1;
            // canvas[(idx+nx)+2] = 1;
            // canvas[(idx+nx)+3] = 1;

            // canvas[(idx-nx)+0] = 1;
            // canvas[(idx-nx)+1] = 1;
            // canvas[(idx-nx)+2] = 1;
            // canvas[(idx-nx)+3] = 1;

            if (lines) {
                cc.bres.plotLine(@floatToInt(i32, oldpt[0]), @floatToInt(i32, oldpt[1]), @floatToInt(i32, x), @floatToInt(i32, y));
                oldpt = .{ x, y };
            }
            // if (lines) |_| {
            // if (i>=1){
            // }}

        }
    }

    // try im.saveU8AsTGA(canvas, @intCast(u16,ny), @intCast(u16,nx), picname);
    try im.saveF32AsTGAGreyNormed(canvas, @intCast(u16, ny), @intCast(u16, nx), picname);
}

pub fn idxsortFn(zvals: []f32, lhs: u32, rhs: u32) bool {
    if (zvals[lhs] < zvals[rhs]) return true else return false;
}

// inplace sorting of faces by z-position
pub fn faceZOrder(vertices: []Vec3, faces: [][4]u32) ![]u32 {
    const zpos = try allocator.alloc(f32, faces.len);
    defer allocator.free(zpos);
    const indices = try allocator.alloc(u32, faces.len);
    // defer allocator.free(indices);

    for (faces) |f, i| {
        const zval = (vertices[f[0]][0] + vertices[f[1]][0] + vertices[f[2]][0] + vertices[f[3]][0]) * 0.25; //Vec3{0.25,0.25,0.25};
        zpos[i] = zval;
        indices[i] = @intCast(u32, i);
    }

    // print("inds[..10]={d}\n",.{indices[0..10].*});
    std.sort.sort(u32, indices, zpos, idxsortFn);
    // print("inds[..10]={d}\n",.{indices[0..10].*});
    return indices;
}

// ignore Z-dim and faces for now... just plotvertices and edges using XY
pub fn drawMeshXY(surf: Mesh, picname: []const u8) !Img2D(f32) {

    // prepare canvas
    var nx: u32 = 2880;
    var ny: u32 = 1800;
    // var canvas = try allocator.alloc(u8,4*nx*ny);
    // for (canvas) |*v,i| v.* = if (i%4==3) 255 else 0;

    const canvas = try Img2D(f32).init(nx, ny);

    // var canvas = try allocator.alloc(f32,nx*ny);
    // for (canvas) |*v,i| v.* = 0; //if (i%4==3) 255 else 0;
    cc.bres.init(&canvas.img[0], &ny, &nx);

    const verts = try allocator.alloc([2]f32, surf.vs.len);
    defer allocator.free(verts);

    // Compute bounds
    const bds = bounds3(surf.vs);
    // const mid = (bds[0] + bds[1]) * Vec3{0.5,0.5,0.5};
    const width = (bds[1] - bds[0]);

    // fill `verts` with pixel positions (f32)
    for (surf.vs) |v, i| {
        const v2 = (v - bds[0]) / width * Vec3{ 0, @intToFloat(f32, ny) - 20, @intToFloat(f32, nx) - 20 } + Vec3{ 0, 10, 10 }; // normalize to [0,ny] [0,nx]
        const x = v2[2];
        const y = v2[1];
        verts[i] = .{ x, y };
    }

    drawEdges(canvas, verts, surf.es);
    try im.saveF32Img2D(canvas, picname);

    return canvas;
}

// test "drawing. make a grid lattice in B&W and color" {
//   // try mkdirIgnoreExists("BlackAndWhiteLattice");
//   const gs = try gridMesh(100,100);
//   defer gs.deinit();

//   const dt = Vec3{0,0,0.1};
//   for (gs.vs) |_,i| {
//     gs.vs[i] += geo.randNormalVec3() * dt;
//   }
//   const img = try drawMeshXY(gs, try join(test_home,"latticemesh_img.tga"));
//   defer img.deinit();

//   try drawMesh3DMovie2(gs, try join(test_home, "BlackAndWhiteLattice/img"));
//   var imgu8 = try allocator.alloc(u8,img.img.len);
//   defer allocator.free(imgu8);

//   for (imgu8) |_,i| imgu8[i] = if (img.img[i]>0) 1 else 0;
//   const res = try fillImg2D(imgu8,img.nx,img.ny);
//   defer toys.allocator.free(res.img);

//   for (res.img) |*v| v.* = v.* % 999;
//   const res2 = try randomColorLabels(allocator,u16,res.img);
//   defer allocator.free(res2);

//   try im.saveU8AsTGA(res2,@intCast(u16,img.ny),@intCast(u16,img.nx), try join(test_home,"surface_colored.tga"));
// }

pub fn drawFaces(img: Img2D(f32), verts: [][2]f32, faces: [][4]u32, _indices: ?[]u32, plotedges: bool) !void {

    // const indices = try faceZOrder(verts,faces);
    var indices = if (_indices) |v| v else blk: {
        var x = try allocator.alloc(u32, faces.len);
        for (x) |*v, i| v.* = @intCast(u32, i);
        break :blk x;
    };

    var nx = img.nx;
    var ny = img.ny;
    cc.bres.init(&img.img[0], &ny, &nx);

    outer: for (indices) |ind| {
        const f = faces[ind];
        var quad: [4][2]u32 = undefined;
        for (quad) |*q, j| {
            const v = verts[f[j]];
            const xi32 = @floatToInt(i32, @floor(v[0]));
            const yi32 = @floatToInt(i32, @floor(v[1]));
            if (!inbounds(img, [2]i32{ xi32, yi32 })) continue :outer; // V2i32{@floatToInt(i32,v[0]) , @floatToInt(i32,v[1])} )
            const x = @floatToInt(u32, @floor(v[0]));
            const y = @floatToInt(u32, @floor(v[1]));
            q.* = [2]u32{ x, y };
        }

        F32COLOR = @intToFloat(f32, (ind + 1) % 2);
        // F32COLOR = @intToFloat(f32,ind);
        renderFilledQuad(img, quad);

        // now plot face edges
        if (plotedges) {
            cc.bres.plotLine(@intCast(i32, quad[0][0]), @intCast(i32, quad[0][1]), @intCast(i32, quad[1][0]), @intCast(i32, quad[1][1]));
            cc.bres.plotLine(@intCast(i32, quad[1][0]), @intCast(i32, quad[1][1]), @intCast(i32, quad[2][0]), @intCast(i32, quad[2][1]));
            cc.bres.plotLine(@intCast(i32, quad[2][0]), @intCast(i32, quad[2][1]), @intCast(i32, quad[3][0]), @intCast(i32, quad[3][1]));
            cc.bres.plotLine(@intCast(i32, quad[3][0]), @intCast(i32, quad[3][1]), @intCast(i32, quad[0][0]), @intCast(i32, quad[0][1]));
        }
    }
}

pub fn drawEdges(img: Img2D(f32), verts: [][2]f32, edges: [][2]u32) void {
    var nx = img.nx;
    var ny = img.ny;
    cc.bres.init(&img.img[0], &ny, &nx);

    for (edges) |e| {
        const v0 = verts[e[0]];
        const v1 = verts[e[1]];
        const x0 = @floatToInt(i32, v0[0]);
        const y0 = @floatToInt(i32, v0[1]);
        const x1 = @floatToInt(i32, v1[0]);
        const y1 = @floatToInt(i32, v1[1]);
        cc.bres.plotLine(x0, y0, x1, y1);
    }
}

// assumes the
pub fn drawVerts(img: Img2D(f32), verts: [][2]f32) void {

    // DRAW PIXELS INTO THE IMAGE BUFFER
    var nx = img.nx;
    var ny = img.ny;
    cc.bres.init(&img.img[0], &ny, &nx);

    // vertices
    for (verts) |v| {
        // const v2 = (v-bds[0]) / width * Vec3{0 , @intToFloat(f32,ny) - 20 , @intToFloat(f32,nx) - 20} + Vec3{0,10,10}; // normalize to [0,ny] [0,nx]
        const x = @floatToInt(i32, v[0]);
        const y = @floatToInt(i32, v[1]);
        cc.bres.setPixel(x, y);
        cc.bres.setPixel(x, y + 1);
        cc.bres.setPixel(x + 1, y);
        cc.bres.setPixel(x + 1, y + 1);
    }
}

pub fn myabs(a: anytype) @TypeOf(a) {
    if (a < 0) return -a else return a;
}

// Render a filled triangle. tri in X,Y coords.
fn renderFilledTri(_img: Img2D(f32), _tri: [3][2]u32) void {

    // var flatBot = tri;
    // var flatTop = tri;

    var img = _img.img;
    var nx = _img.nx;
    // var ny = _img.ny;

    // var i:u32 = 0;
    // var j:u32 = 0;

    // sort triangle
    var a = uvec2(_tri[0]);
    var b = uvec2(_tri[1]);
    var c = uvec2(_tri[2]);

    // sort by y
    var temp = a;
    if (a[1] > b[1]) {
        temp = b;
        b = a;
        a = temp;
    } // sort 0,1
    if (b[1] > c[1]) {
        temp = c;
        c = b;
        b = temp;
    } // sort 1,2
    if (a[1] > b[1]) {
        temp = b;
        b = a;
        a = temp;
    } // sort 0,1

    // if a==b then either draw point or line from a==b->c
    if (a[0] == b[0] and a[1] == b[1]) {
        if (a[0] == c[0] and a[1] == c[1]) {
            const x = @floatToInt(u32, @floor(a[0]));
            const y = @floatToInt(u32, @floor(a[1]));
            img[y * nx + x] = 1; // Set Pixel
            return;
        } else {
            return;
            // drawline(_img,a,b);
        }
    }

    // if b==c then draw line from a->b==c.
    if (c[0] == b[0] and c[1] == b[1]) {
        return;
    }

    // if aligned in X then just draw vertical line
    if (a[0] == b[0] and b[0] == c[0]) {
        var y = @floatToInt(u32, @floor(a[1]));
        const x = @floatToInt(u32, @floor(a[0]));
        const ymax = @floatToInt(u32, @floor(c[1]));
        while (y < ymax) : (y += 1) {
            img[y * nx + x] = 1;
        }
        return;
    }

    // if aligned in Y then just draw horizontal line
    const xmin = @floatToInt(u32, @floor(min3(a[0], b[0], c[0])));
    const xmax = @floatToInt(u32, @floor(max3(a[0], b[0], c[0])));
    if (a[1] == b[1] and b[1] == c[1]) {
        const y = @floatToInt(u32, @floor(a[1]));
        var x = xmin;
        while (x < xmax) : (x += 1) {
            img[y * nx + x] = 1;
        }
        return;
    }

    // if flat on low-y side.
    if (a[1] == b[1]) {
        if (a[0] > b[0]) {
            temp = b;
            b = a;
            a = temp;
        } // swap a,b
        fillFlatBaseTri(_img, c, a, b);
        return;
    }

    // if flat on high-y side.
    if (b[1] == c[1]) {
        if (b[0] > c[0]) {
            temp = c;
            c = b;
            b = temp;
        } // sort 1,2
        fillFlatBaseTri(_img, a, b, c);
        return;
    }

    // find the midpoint
    // const dv = c-a; // hi-lo
    // const dw = b-a;
    // midpoint = a + alpha * dv; midpoint.y = b.y;
    // alpha = (midpoint.y - a.y) / dv.y
    const dyRatio = (b[1] - a[1]) / (c[1] - a[1]); // guaranteed to be 0<x<1;
    const xmid = a[0] + dyRatio * (c[0] - a[0]);

    // sort out left and right side of base
    var left = b;
    var right = vec2(.{ @floor(xmid), b[1] });
    if (left[0] == right[0]) return; // TODO FIXME

    // print("all the triangles...\n" , .{});
    // print("a={d}\n",.{a});
    // print("b={d}\n",.{b});
    // print("c={d}\n",.{c});
    // print("left={d}\n",.{left});
    // print("right={d}\n",.{right});

    // if (abs2(cross2(left-a,right-a)) < 0) {const temp2=left; left=right; right=temp2;}
    if (left[0] > right[0]) {
        const temp2 = left;
        left = right;
        right = temp2;
    }

    // now from bottom up both legs
    fillFlatBaseTri(_img, a, left, right);
    fillFlatBaseTri(_img, c, left, right);
}

fn renderFilledQuad(_img: Img2D(f32), _quad: [4][2]u32) void {
    renderFilledTri(_img, _quad[0..3].*);
    renderFilledTri(_img, .{ _quad[2], _quad[3], _quad[0] });
}

test "drawing. render filled tri and quad" {
    // pub fn main() !void {
    var img = try Img2D(f32).init(100, 100);
    defer img.deinit();
    const tri = .{ .{ 0, 0 }, .{ 10, 50 }, .{ 50, 0 } };
    renderFilledTri(img, tri);

    img.img[0 * img.nx + 0] = 2;
    img.img[50 * img.nx + 10] = 2;
    img.img[0 * img.nx + 50] = 2;

    try im.saveF32Img2D(img, test_home ++ "filledtri.tga");

    const quad = .{ .{ 60, 60 }, .{ 65, 60 }, .{ 65, 65 }, .{ 60, 65 } };
    renderFilledQuad(img, quad);

    img.img[60 * img.nx + 60] = 2;
    img.img[65 * img.nx + 60] = 2;
    img.img[60 * img.nx + 65] = 2;
    img.img[65 * img.nx + 65] = 2;

    try im.saveF32Img2D(img, test_home ++ "filledquad.tga");
}

/// Fill triangle with flat base or roof.
/// Assume a,b,c are already sorted s.t. (b-a) x (c-a) > 0;
/// and that b.y == c.y
/// this means we fill horizontal rows from a.y -> (b.y==c.y)
/// and we fill from left (b) to right (c)
/// b==left c==right
pub fn fillFlatBaseTri(_img: Img2D(f32), a: Vec2, left: Vec2, right: Vec2) void {

    // print("a={d}\n",.{a});
    // print("left={d}\n",.{left});
    // print("right={d}\n",.{right});
    // assert(abs2(cross2(left-a,right-a)) > 0);

    assert(left[0] < right[0]);
    assert(left[1] == right[1]);

    const img = _img.img;
    const nx = _img.nx;

    const inc: i2 = if (left[1] > a[1]) 1 else -1;

    // var dy = @floatToInt(u32 , myabs(left[1] - a[1]));
    // assert(dy!=0);
    // dy = if (dy>0) dy else -dy; // abs(dy)

    // const dxLeft  = @intToFloat(f32, @intCast(i33,left[0]) -a[0]) / @intToFloat(f32,dy);
    // const dxRight = @intToFloat(f32, @intCast(i33,right[0])-a[0]) / @intToFloat(f32,dy);
    const dxLeft = (left[0] - a[0]) / myabs(left[1] - a[1]);
    const dxRight = (right[0] - a[0]) / myabs(right[1] - a[1]);

    const ytarget = @floatToInt(u32, left[1]);
    var y = @floatToInt(i32, a[1]);
    var dl = @as(f32, 0);
    var dr = @as(f32, 0);

    // print("ytarget = {}\n" , .{ytarget});

    while (true) : (y += inc) {
        // const dy2 = y-a[1];
        const x0 = @floatToInt(u32, a[0] + dl);
        dl += dxLeft;
        const x1 = @floatToInt(u32, a[0] + dr);
        dr += dxRight;
        // print("x0,x1,y = {d},{d},{d}\n" , .{x0,x1,y});
        var _x = x0;
        while (_x <= x1) : (_x += 1) img[@intCast(usize, y) * nx + _x] = F32COLOR;
        if (y == ytarget) break;
    }
}

var F32COLOR: f32 = 1.0;
