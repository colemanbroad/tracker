// GEOMETRY  GEOMETRY  GEOMETRY  GEOMETRY
// GEOMETRY  GEOMETRY  GEOMETRY  GEOMETRY
// GEOMETRY  GEOMETRY  GEOMETRY  GEOMETRY

const std = @import("std");

const print = std.debug.print;
const assert = std.debug.assert;
const expect = std.testing.expect;

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

var allocator = std.testing.allocator;
// const print = std.debug.print;

// const Vector = std.meta.Vector;
// const Vector = s
pub const Vec3 = @Vector(3, f32);

test {
    std.testing.refAllDecls(@This());
}

pub fn print2(
    comptime src_info: std.builtin.SourceLocation,
    comptime fmt: []const u8,
    args: anytype,
) void {
    if (true) return;
    const s1 = comptime std.fmt.comptimePrint("{s}:{d}:{d} ", .{ src_info.file, src_info.line, src_info.column });
    std.debug.print(s1[41..] ++ fmt, args);
}

// mean 0 stddev 1
pub fn randNormalVec3() Vec3 {
    return Vec3{ random.floatNorm(f32), random.floatNorm(f32), random.floatNorm(f32) };
}

pub fn randNormalVec2() Vec2 {
    return Vec2{ random.floatNorm(f32), random.floatNorm(f32) };
}

// spiral walk around the unit sphere
pub fn sphereTrajectory() [100]Vec3 {
    var phis: [100]f32 = undefined;
    // for (phis) |*v,i| v.* = ((@intToFloat(f32,i)+1)/105) * pi;
    for (&phis, 0..) |*v, i| v.* = ((@as(f32, @floatFromInt(i))) / 99) * pi;
    var thetas: [100]f32 = undefined;
    // for (thetas) |*v,i| v.* = ((@intToFloat(f32,i)+1)/105) * 2*pi;
    for (&thetas, 0..) |*v, i| v.* = ((@as(f32, @floatFromInt(i))) / 99) * 2 * pi;
    var pts: [100]Vec3 = undefined;
    for (&pts, 0..) |*v, i| v.* = Vec3{ @cos(phis[i]), @sin(thetas[i]) * @sin(phis[i]), @cos(thetas[i]) * @sin(phis[i]) }; // ZYX coords
    return pts;
}

const Allocator = std.mem.Allocator;

const Pix = @Vector(2, u31);
pub fn pt2PixCast(p: Vec2) Pix {
    return Pix{ @as(u31, @intFromFloat(p[0])), @as(u31, @intFromFloat(p[1])) };
}

pub fn newBBox(x0: f32, x1: f32, y0: f32, y1: f32) BBox {
    return BBox{ .x = .{ .lo = x0, .hi = x1 }, .y = .{ .lo = y0, .hi = y1 } };
}

pub fn affine(_p: Vec2, bb0: BBox, bb1: BBox) Vec2 {
    var p = _p;

    p -= Vec2{ bb0.x.lo, bb0.y.lo };
    p *= Vec2{ (bb1.x.hi - bb1.x.lo) / (bb0.x.hi - bb0.x.lo), (bb1.y.hi - bb1.y.lo) / (bb0.y.hi - bb0.y.lo) };
    p += Vec2{ bb1.x.lo, bb1.y.lo };

    return p;
}

pub fn swap(comptime T: type, a: *T, b: *T) void {
    const temp = a.*;
    a.* = b.*;
    b.* = temp;
}

pub const Range = struct { hi: f32, lo: f32 };
pub const BBox = struct { x: Range, y: Range };

// memory layout: start at top left. first go down columns, then right across rows.
pub const Mat3x3 = [9]f32;

pub fn vec3(a: [3]f32) Vec3 {
    return Vec3{ a[0], a[1], a[2] };
}

pub fn invert3x3(mat: Mat3x3) Mat3x3 {
    const v1 = vec3(mat[0..3].*);
    const v2 = vec3(mat[3..6].*);
    const v3 = vec3(mat[6..9].*);
    const v1v2 = cross(v1, v2);
    const v2v3 = cross(v2, v3);
    const v3v1 = cross(v3, v1);
    const d = dot(v1, v2v3);
    return Mat3x3{
        v2v3[0] / d,
        v3v1[0] / d,
        v1v2[0] / d,
        v2v3[1] / d,
        v3v1[1] / d,
        v1v2[1] / d,
        v2v3[2] / d,
        v3v1[2] / d,
        v1v2[2] / d,
    };
}

pub fn matFromVecs(v0: [3]f32, v1: [3]f32, v2: [3]f32) Mat3x3 {
    return Mat3x3{ v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2] };
}

test "geometry. matrix inverse" {
    // const mA = [9]f32{1,2,0,0,1,0,0,0,1}; // checked against julia
    const mA = [9]f32{ 1, 2, -9, 4, 1, -2, 3, 3, 0 }; // checked against julia
    print2(@src(), "\n\n{d}\n\n", .{invert3x3(mA)});
}

// Find the intersection points between a line and an axis-aligned bounding box.
// NOTE: ray.pt0 is the starting point and ray.pt1 is any other point along the line.
// TODO: handle case where ray.pt0 is inside the bounding box.
pub fn intersectRayAABB(ray: Ray, box: Ray) struct { pt0: ?Vec3, pt1: ?Vec3 } {
    assert(box.pt0[0] < box.pt1[0]);
    assert(box.pt0[1] < box.pt1[1]);
    assert(box.pt0[2] < box.pt1[2]);

    // Compute alpha for each of the six orthogonal planes.
    // intersection_point_i = alpha_i * dv + pt0
    const dr = ray.pt1 - ray.pt0;
    const rp = ray.pt0;

    // alphasNear is a vector of scalar multipliers. Each scalar in alphasNear determines an intersection
    // point with an orthogonal X,Y or Z plane. But we still do not know if that point lies _inside_ the face
    // of our box. NOTE: near / far refer only to distance to the origin, not to starting ray.pt0 .
    const alphasNear = (box.pt0 - rp) / dr;
    const alphasFar = (box.pt1 - rp) / dr;

    // Compute each of the six intersection points.
    const ipZNear = rp + dr * @as(Vec3, @splat(alphasNear[0]));
    const ipYNear = rp + dr * @as(Vec3, @splat(alphasNear[1]));
    const ipXNear = rp + dr * @as(Vec3, @splat(alphasNear[2]));
    const ipZFar = rp + dr * @as(Vec3, @splat(alphasFar[0]));
    const ipYFar = rp + dr * @as(Vec3, @splat(alphasFar[1]));
    const ipXFar = rp + dr * @as(Vec3, @splat(alphasFar[2]));

    // Test if each of the six intersection points lies inside the rectangular box face
    const p0 = box.pt0;
    const p1 = box.pt1;
    const ipZNearTest = p0[1] <= ipZNear[1] and ipZNear[1] < p1[1] and p0[2] <= ipZNear[2] and ipZNear[2] < p1[2];
    const ipYNearTest = p0[0] <= ipYNear[0] and ipYNear[0] < p1[0] and p0[2] <= ipYNear[2] and ipYNear[2] < p1[2];
    const ipXNearTest = p0[0] <= ipXNear[0] and ipXNear[0] < p1[0] and p0[1] <= ipXNear[1] and ipXNear[1] < p1[1];
    const ipZFarTest = p0[1] <= ipZFar[1] and ipZFar[1] < p1[1] and p0[2] <= ipZFar[2] and ipZFar[2] < p1[2];
    const ipYFarTest = p0[0] <= ipYFar[0] and ipYFar[0] < p1[0] and p0[2] <= ipYFar[2] and ipYFar[2] < p1[2];
    const ipXFarTest = p0[0] <= ipXFar[0] and ipXFar[0] < p1[0] and p0[1] <= ipXFar[1] and ipXFar[1] < p1[1];

    // print2(@src(),"Our box test results are:\n\n",.{});
    // print2(@src(),"pt1= {d} intersection?: {}\n",.{ipZNear, ipZNearTest});
    // print2(@src(),"pt2= {d} intersection?: {}\n",.{ipYNear, ipYNearTest});
    // print2(@src(),"pt3= {d} intersection?: {}\n",.{ipXNear, ipXNearTest});
    // print2(@src(),"pt4= {d} intersection?: {}\n",.{ipZFar, ipZFarTest});
    // print2(@src(),"pt5= {d} intersection?: {}\n",.{ipYFar, ipYFarTest});
    // print2(@src(),"pt6= {d} intersection?: {}\n",.{ipXFar, ipXFarTest});

    var pIn: ?Vec3 = null;
    var pOut: ?Vec3 = null;
    var alphaIn: ?f32 = null;
    var alphaOut: ?f32 = null;

    if (ipZNearTest) {
        pIn = ipZNear;
        alphaIn = alphasNear[0];
    }
    if (ipYNearTest) {
        pIn = ipYNear;
        alphaIn = alphasNear[1];
    }
    if (ipXNearTest) {
        pIn = ipXNear;
        alphaIn = alphasNear[2];
    }
    if (ipZFarTest) {
        pOut = ipZFar;
        alphaOut = alphasFar[0];
    }
    if (ipYFarTest) {
        pOut = ipYFar;
        alphaOut = alphasFar[1];
    }
    if (ipXFarTest) {
        pOut = ipXFar;
        alphaOut = alphasFar[2];
    }

    if (alphaIn) |a| {
        if (alphaOut) |b| {
            assert(a >= 0);
            assert(b >= 0);

            if (b <= a) {
                // print2(@src(),"starting pIn,pOut = {d:0.2} , {d:0.2}\n" , .{pIn,pOut});
                const temp = pIn;
                pIn = pOut;
                pOut = temp;
                // print2(@src(),"swapped  pIn,pOut = {d:0.2} , {d:0.2}\n" , .{pIn,pOut});
            }
        }
    }

    return .{ .pt0 = pIn, .pt1 = pOut };

    // if (ipZNearTest or ipYNearTest or ipXNearTest or ipZFarTest  or ipYFarTest  or ipXFarTest) {return Ray{.pt0=pIn , .pt1=pOut};}
    // else {return null;}
}

test "geometry. Axis aligned bounding box intersection test" {
    // fn testAxisAlignedBoundingBox() !void {
    print2(@src(), "\n", .{});
    {
        // The ray intersects the box at the points {0,0,0} and {1,3,1}
        const ray = Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = .{ 1, 3, 1 } };
        const box = Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = .{ 3, 3, 3 } };
        const res = intersectRayAABB(ray, box);
        try expect(l2norm(res.pt0.? - Vec3{ 0, 0, 0 }) < 1e-6);
        try expect(l2norm(res.pt1.? - Vec3{ 1, 3, 1 }) < 1e-6);
        print2(@src(), "{d}\n", .{res});
        print2(@src(), "\n", .{});
    }

    {
        // The ray doesn't intersect the box at all
        const ray = Ray{ .pt0 = .{ -1, 0, 0 }, .pt1 = .{ -1, 3, 1 } };
        const box = Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = .{ 3, 3, 3 } };
        const res = intersectRayAABB(ray, box);
        print2(@src(), "{d}\n", .{res});
        try expect(res.pt0 == null and res.pt1 == null);
        print2(@src(), "\n", .{});
    }

    {
        // The ray intersects the box at a single point {0,0,0}
        const ray = Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = .{ -1, 3, 1 } };
        const box = Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = .{ 3, 3, 3 } };
        const res = intersectRayAABB(ray, box);
        try expect((res.pt0 == null) != (res.pt1 == null)); // `!=` can be used as `xor`
        print2(@src(), "{d}\n", .{res});
        print2(@src(), "\n", .{});
    }
}

pub const Ray = struct {
    pt0: Vec3,
    pt1: Vec3,
};

pub const Face = struct {
    pt0: Vec3,
    pt1: Vec3,
    pt2: Vec3,
};

pub const AxisFace = struct {
    pt0: Vec3,
    axis: Axis,
};

pub const Axis = enum(u2) { z, y, x };

// Inverts a 3x3 matrix to find the exact intersection point between a Ray
// and an arbitrary triangluar face in 3D space. `b` and `c` are scalar values
// defined by `b*(face.pt1-face.pt0) + c*(face.pt2-face.pt0) + face.pt0 = pt` (intersection point).
pub fn intersectRayFace(ray: Ray, face: Face) struct { b: f32, c: f32, pt: Vec3 } {

    //
    const dr = ray.pt1 - ray.pt0;
    const r0 = ray.pt0;
    const p0 = face.pt0;
    const dp1 = face.pt1 - face.pt0;
    const dp2 = face.pt2 - face.pt0;

    // Solve r0-p0 = [[dr dp1 dp2]]*[a b c]
    const matrix = matFromVecs(dr, dp1, dp2);
    const matinv = invert3x3(matrix);
    const abc = matVecMul(matinv, r0 - p0);

    const intersectionPoint1 = @as(Vec3, @splat(abc[0])) * dr + r0;
    const intersectionPoint2 = @as(Vec3, @splat(abc[1])) * dp1 + @as(Vec3, @splat(abc[2])) * dp2 + p0;

    print2(@src(), "\n\nabc={d}\n\n", .{abc});
    print2(@src(), "intersection point (v1) = a*dr + r0 = {d} \n", .{intersectionPoint1});
    print2(@src(), "intersection point (v2) = b*dp1 + c*dp2 + p0 = {d} \n", .{intersectionPoint2});

    return .{ .b = abc[1], .c = abc[2], .pt = intersectionPoint2 };
}

test "geometry. Intersect Ray with (arbitrary) Face" {
    const r1 = Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = .{ 1, 1.5, 1.5 } };
    const f1 = Face{ .pt0 = .{ 5, 0, 0 }, .pt1 = .{ 0, 5, 0 }, .pt2 = .{ 0, 0, 5 } };
    const x0 = intersectRayFace(r1, f1);
    print2(@src(), "\n{d}\n", .{x0});
}

// (Where) does a ray intersect with a quadralateral face?
// Uses ZYX convention.
pub fn intersectRayAxisFace(ray: Ray, face: AxisFace) ?Vec3 {
    assert(false); // INCOMPLETE FUNCTION. WIP.

    // Compute alpha for each of the three orthogonal intersection planes.
    // intersection_point_i = alpha_i * dv + pt0
    const dv = ray.pt1 - ray.pt0;
    const rp = ray.pt0;
    const fp = face.pt0;

    const alphas = (fp - rp) / dv;
    // const a_low  = (box.low - pt0) / dv;
    // const a_hi   = (box.hi  - pt0) / dv;
    const xZ = rp + dv * @as(Vec3, @splat(alphas[0]));
    const xY = rp + dv * @as(Vec3, @splat(alphas[1]));
    const xX = rp + dv * @as(Vec3, @splat(alphas[2]));

    // var res:?Vec3 = null;
    switch (face.axis) {
        .z => if (0 <= rp[2] and rp[2] <= fp[2] and 0 <= rp[1] and rp[1] <= fp[1]) return xZ,
        .y => if (0 <= rp[2] and rp[2] <= fp[2] and 0 <= rp[0] and rp[0] <= fp[0]) return xY,
        .x => if (0 <= rp[0] and rp[0] <= fp[0] and 0 <= rp[1] and rp[1] <= fp[1]) return xX,
    }

    return null;
}

pub fn dot2(a: Vec2, b: Vec2) f32 {
    return a[0] * b[0] + a[1] * b[1];
}

pub fn pointInTriangle2d(pt: Vec2, tri: [3]Vec2) bool {
    // get circumcircle from triangle points
    // const a = tri[1] - tri[0]; // cross2(tri[0],pt);
    // const b = tri[2] - tri[1]; // cross2(tri[1],pt);
    // const c = tri[0] - tri[2]; // cross2(tri[2],pt);
    // const v = pt - tri[0];
    const xa = cross2(tri[1] - tri[0], tri[1] - pt);
    const xb = cross2(tri[2] - tri[1], tri[2] - pt);
    const xc = cross2(tri[0] - tri[2], tri[0] - pt);
    // @breakpoint();
    if ((xa > 0 and xb > 0 and xc > 0) or (xa < 0 and xb < 0 and xc < 0)) return true else return false;
}

pub const CircleR2 = struct { pt: Vec2, r2: f32 };

// see [matrix formula](https://en.wikipedia.org/wiki/Circumscribed_circle)
pub fn getCircumcircle2d(tri: [3]Vec2) CircleR2 {

    // normalize for numerical reasons
    const center = (tri[0] + tri[1] + tri[2]) / Vec2{ 3.0, 3.0 };

    // const center = Vec2{0.0,0.0};
    // const tri2 = [3]Vec2{tri[0]-center , tri[1]-center , tri[2]-center};

    const _a = (tri[0] - center);
    const _b = (tri[1] - center);
    const _c = (tri[2] - center);
    const mindist = @min(abs2(_a), abs2(_b), abs2(_c)) * 0.1;
    // const mindist = 1.0;

    const a = _a / Vec2{ mindist, mindist };
    const b = _b / Vec2{ mindist, mindist };
    const c = _c / Vec2{ mindist, mindist };

    const a2 = dot2(a, a);
    const b2 = dot2(b, b);
    const c2 = dot2(c, c);

    const sx = 0.5 * det(Mat3x3{ a2, b2, c2, a[1], b[1], c[1], 1, 1, 1 });
    const sy = 0.5 * det(Mat3x3{ a[0], b[0], c[0], a2, b2, c2, 1, 1, 1 });
    const m = det(Mat3x3{ a[0], b[0], c[0], a[1], b[1], c[1], 1, 1, 1 });
    const n = det(Mat3x3{ a[0], b[0], c[0], a[1], b[1], c[1], a2, b2, c2 });

    // compute circumcenter
    const centerpoint = Vec2{ sx / m, sy / m } + center;
    const radiusSquared = (n / m + (sx * sx + sy * sy) / (m * m)) * mindist * mindist;

    // print2(@src(),"CircumCircle\n",.{});
    // print2(@src(),"a={d}\n",.{a});
    // print2(@src(),"b={d}\n",.{b});
    // print2(@src(),"c={d}\n",.{c});
    // print2(@src(),"center={d}\n",.{center});
    // print2(@src(),"radiusSquared={d}\n\n\n",.{radiusSquared});

    return .{ .pt = centerpoint, .r2 = radiusSquared };
}

pub fn getCircumcircle2dv2(tri: [3]Vec2) CircleR2 {
    const d01 = (tri[1] - tri[0]) / Vec2{ 2, 2 };
    // const d12 = (tri[2] - tri[1]) / Vec2{2,2};
    const d20 = (tri[0] - tri[2]) / Vec2{ 2, 2 };
    const x0 = tri[0] + d01; // side midpoints
    const x1 = tri[2] + d20; // side midpoints
    // [0 -1 1 0]*[x y]
    const dx0 = Vec2{ -d01[1], d01[0] };
    // const rot90_d12 = Vec2{-d12[1],d12[0]};
    const dx1 = Vec2{ -d20[1], d20[0] };

    // setup and solve Aw=b
    // [[dx0] [-dx1]][w0;w1] = [x1-x0]
    const b = x1 - x0;
    const mA = [4]f32{ dx0[0], dx0[1], -dx1[0], -dx1[1] };
    const mAinv = inv2x2(f32, mA);
    const w = mul2x2MatVec(f32, mAinv, b);

    const circumcenter = x0 + Vec2{ w[0] * dx0[0], w[0] * dx0[1] };
    // const circumcenter2 = x1 + Vec2{w[1]*dx1[0],w[1]*dx1[1]};
    // print2(@src(),"centers:\n",.{});
    // print2(@src(),"0:     {d}\n",.{circumcenter});
    // print2(@src(),"1:     {d}\n",.{circumcenter2});
    // print2(@src(),"delta: {d}\n",.{circumcenter2 - circumcenter});

    // intersections
    // const p_ab = intersection(tri[0]+d01 , tri[0]+d01+rot90_d01 , tri[1]+d12 , tri[1]+d12+rot90_d12);
    const r0 = tri[0] - circumcenter;
    // const r1 = tri[1]-circumcenter;
    // const r2 = tri[2]-circumcenter;
    // print2(@src(),"Radii: {d} .. {d} .. {d} \n" , .{r0,r1,r2});
    const ret = .{ .pt = circumcenter, .r2 = dot2(r0, r0) };
    return ret;
}

pub fn inv2x2(comptime T: type, mat: [4]T) [4]T {
    const d = mat[0] * mat[3] - mat[1] * mat[2];
    const mat2 = .{ mat[3] / d, -mat[1] / d, -mat[2] / d, mat[0] / d };
    return mat2;
}

pub fn mul2x2MatVec(comptime T: type, mat: [4]T, vec: [2]T) [2]T {
    const r0 = mat[0] * vec[0] + mat[2] * vec[1];
    const r1 = mat[1] * vec[0] + mat[3] * vec[1];
    const mat2 = .{ r0, r1 };
    return mat2;
}

pub fn det(mat: Mat3x3) f32 {
    const a = Vec3{ mat[0], mat[1], mat[2] };
    const b = Vec3{ mat[3], mat[4], mat[5] };
    const c = Vec3{ mat[6], mat[7], mat[8] };
    return dot(a, cross(b, c)); // determinant of 3x3 matrix is equal to scalar triple product
}

pub fn pointInTriangleCircumcircle2d(pt: Vec2, tri: [3]Vec2) bool {
    const circumcircle = getCircumcircle2dv2(tri);
    const p_center = circumcircle.pt;
    const r2 = circumcircle.r2;
    const delta = pt - p_center;
    if (dot2(delta, delta) <= r2) return true else return false;
}

test "geometry. test point in circumcircle" {
    const t1 = Vec2{ 0, 0 };
    const t2 = Vec2{ 0, 5 };
    const t3 = Vec2{ 5, 0 };
    const tri = [3]Vec2{ t1, t2, t3 };

    const v1 = Vec2{ 1, 1 }; // true
    const v2 = Vec2{ -1, -1 }; // false

    // outside of the triangle , but inside the circumcircle
    const _d3 = @sqrt(2.0) * 5.0 / 2.0 + 1.0;
    const v3 = Vec2{ _d3, _d3 }; // true

    print2(@src(), "{b}\n", .{pointInTriangleCircumcircle2d(v1, tri)});
    print2(@src(), "{b}\n", .{pointInTriangleCircumcircle2d(v2, tri)});
    print2(@src(), "{b}\n", .{pointInTriangleCircumcircle2d(v3, tri)});
}

test "geometry. fuzz circumcircle" {
    var count: u32 = 0;

    count = 0;
    while (count < 100) : (count += 1) {
        const radius = random.float(f32) * 100;
        const x0 = random.float(f32) * 100.0;
        const y0 = random.float(f32) * 100.0;
        const cp = Vec2{ x0, y0 }; // centerpoint
        const r = Vec2{ radius, radius }; // radius

        var p0 = cp + r * normalize2(randNormalVec2());
        var p1 = cp + r * normalize2(randNormalVec2());
        var p2 = cp + r * normalize2(randNormalVec2());
        var p3 = cp + r * normalize2(randNormalVec2()) * Vec2{ 1.01, 1.01 };
        print2(@src(), "count {d}  \n", .{count}); // , p0,p1,p2,p3});

        errdefer print2(@src(), "{d}\n{d}\n{d}\n{d}\n", .{ p0, p1, p2, p3 });
        try expect(!pointInTriangleCircumcircle2d(p3, .{ p0, p1, p2 }));
    }

    count = 0;
    while (count < 100) : (count += 1) {
        const radius = random.float(f32) * 100;
        const x0 = random.float(f32) * 100.0;
        const y0 = random.float(f32) * 100.0;
        const cp = Vec2{ x0, y0 }; // centerpoint
        const r = Vec2{ radius, radius }; // radius

        var p0 = cp + r * normalize2(randNormalVec2());
        var p1 = cp + r * normalize2(randNormalVec2());
        var p2 = cp + r * normalize2(randNormalVec2());
        var p3 = cp + r * normalize2(randNormalVec2()) * Vec2{ 0.99, 0.99 };
        print2(@src(), "count {d}  \n", .{count}); // , p0,p1,p2,p3});
        errdefer print2(@src(), "{d}\n{d}\n{d}\n{d}\n", .{ p0, p1, p2, p3 });
        try expect(pointInTriangleCircumcircle2d(p3, .{ p0, p1, p2 }));
    }
}

test "geometry. test point in triange" {
    // pub fn main() void {
    const t1 = Vec2{ 0, 0 };
    const t2 = Vec2{ 0, 5 };
    const t3 = Vec2{ 5, 0 };
    const tri = [3]Vec2{ t1, t2, t3 };

    const v1 = Vec2{ 1, 1 }; // true
    const v2 = Vec2{ -1, -1 }; // false

    print2(@src(), "{b}\n", .{pointInTriangle2d(v1, tri)});
    print2(@src(), "{b}\n", .{pointInTriangle2d(v2, tri)});
}

pub fn lerpVec2(t: f32, a: Vec2, b: Vec2) Vec2 {
    return a + Vec2{ t, t } * (b - a);
}

// VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2
// VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2
// VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2

pub const Ray2 = [2]Vec2;
pub const Vec2 = @Vector(2, f32);

pub fn vec2(a: [2]f32) Vec2 {
    return Vec2{ a[0], a[1] };
}

pub fn uvec2(a: [2]u32) Vec2 {
    return Vec2{ @as(f32, @floatFromInt(a[0])), @as(f32, @floatFromInt(a[1])) };
}

pub fn abs2(a: Vec2) f32 {
    return @sqrt(a[0] * a[0] + a[1] * a[1]);
}

// Requires XYZ order (or some rotation thereof)
pub fn cross2(a: Vec2, b: Vec2) f32 {
    return a[0] * b[1] - a[1] * b[0];
}

// Requires XYZ order (or some rotation thereof)
pub fn cross(a: Vec3, b: Vec3) Vec3 {
    return Vec3{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

test "geometry. cross()" {
    {
        const a = Vec3{ 1, 0, 0 };
        const b = Vec3{ 0, 1, 0 };
        print2(@src(), "\n{d:.3}", .{cross(a, b)});
    }

    {
        const a = Vec3{ 0, 1, 0 };
        const b = Vec3{ 0, 0, 1 };
        print2(@src(), "\n{d:.3}", .{cross(a, b)});
    }

    {
        const a = Vec3{ 0, 1, 0 };
        const b = Vec3{ 0, 0, 1 };
        print2(@src(), "\n{d:.3}", .{cross(a, b)});
    }
}

pub fn dot(a: Vec3, b: Vec3) f32 {
    // return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] ;
    return @reduce(.Add, a * b);
}

pub fn l2norm(a: Vec3) f32 {
    return @sqrt(dot(a, a));
}

pub fn normalize(a: Vec3) Vec3 {
    const s = l2norm(a);
    return a / @as(Vec3, @splat(s));
}

pub fn normalize2(a: Vec2) Vec2 {
    const s = abs2(a);
    return a / @as(Vec2, @splat(s));
}

pub fn scale(a: Vec3, b: f32) Vec3 {
    return a * @as(Vec3, @splat(b));
}

// res = Mat*vec. mat stored as [col1 col2 col3]. Access with mat[3*col + row].
pub fn matVecMul(mat: [9]f32, vec: Vec3) Vec3 {
    return Vec3{
        mat[0] * vec[0] + mat[3] * vec[1] + mat[6] * vec[2],
        mat[1] * vec[0] + mat[4] * vec[1] + mat[7] * vec[2],
        mat[2] * vec[0] + mat[5] * vec[1] + mat[8] * vec[2],
    };
}

// res = Mat*vec. mat stored as [col1 col2 col3]. Access with mat[3*col + row].
pub fn matMatMul(matL: [9]f32, matR: [9]f32) [9]f32 {
    return [9]f32{
        matL[0] * matR[0] + matL[3] * matR[1] + matL[6] * matR[2],
        matL[1] * matR[0] + matL[4] * matR[1] + matL[7] * matR[2],
        matL[2] * matR[0] + matL[5] * matR[1] + matL[8] * matR[2],

        matL[0] * matR[3] + matL[3] * matR[4] + matL[6] * matR[5],
        matL[1] * matR[3] + matL[4] * matR[4] + matL[7] * matR[5],
        matL[2] * matR[3] + matL[5] * matR[4] + matL[8] * matR[5],

        matL[0] * matR[6] + matL[3] * matR[7] + matL[6] * matR[8],
        matL[1] * matR[6] + matL[4] * matR[7] + matL[7] * matR[8],
        matL[2] * matR[6] + matL[5] * matR[7] + matL[8] * matR[8],
    };
}

test "geometry. matrix multiplication" {
    const a = Vec3{ 0.5125063146216244, 0.161090383449368, 0.5436574027867314 };
    const A1 = [9]f32{ 0.943902, 0.775719, 0.931731, 0.0906212, 0.178994, 0.729976, 0.00516308, 0.572436, 0.217663 };
    const A2 = [9]f32{ 0.117929, 0.637452, 0.395997, 0.0014168, 0.442474, 0.450939, 0.970842, 0.382466, 0.57684 };

    print2(@src(), "\n matVecMul : {d} ", .{matVecMul(A1, a)}); // { 0.5011608600616455, 0.7376041412353516, 0.7134442329406738 }
    // From Julia (Float64)
    // 0.5011606467452332
    // 0.7376040111844648
    // 0.7134446669414377

    print2(@src(), "\n matMatMul : {d} ", .{matMatMul(A1, A2)}); // { 0.1711246371269226, 0.4322627782821655, 0.6613966822624207, 0.04376308247447014, 0.3384329378604889, 0.4224681854248047, 0.9540175199508667, 1.151763677597046, 1.3093112707138062 }

    // 0.171124  0.0437631  0.954017
    // 0.432262  0.338433   1.15176
    // 0.661397  0.422469   1.30931
}

// yaw, pitch, and roll angles are α, β and γ
// 9 components in column-first order
pub fn rotYawPitchRoll(yaw: f32, pitch: f32, roll: f32) [9]f32 {
    const ca = @cos(yaw);
    const sa = @sin(yaw);
    const cb = @cos(pitch);
    const sb = @sin(pitch);
    const cg = @cos(roll);
    const sg = @sin(roll);

    return .{
        ca * cb,
        sa * cb,
        -sb,
        ca * sb * sg - sa * cg,
        sa * sb * sg + ca * cg,
        cb * sg,
        ca * sb * cg + sa * sg,
        sa * sb * cg - ca * sg,
        cb * cg,
    };
}

const pi = 3.14159265359;

test "geometry. rotYawPitchRoll()" {
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const rv = normalize(randNormalVec3());
        print2(@src(), "Len {} .. ", .{l2norm(rv)});
        const ypr = randNormalVec3() * Vec3{ 2 * pi, 2 * pi, 2 * pi };
        const rot = rotYawPitchRoll(ypr[0], ypr[1], ypr[2]);
        const rotatedVec = matVecMul(rot, rv);
        print2(@src(), "Len After {} .. \n", .{l2norm(rotatedVec)});
    }
}

// rotate the first argument vc through the rotor defined by vectors va^vb
// rotateCwithRotorAB(x,x,y) == αy
// See https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
// WARNING: components must be in XYZ order for `cross` to work (or a cyclic permutation, i.e. YZX or ZXY).
pub fn rotateCwithRotorAB(vc: Vec3, va: Vec3, vb: Vec3) Vec3 {
    const anorm = normalize(va);
    const bnorm = normalize(vb);
    const k = cross(anorm, bnorm);
    const knorm = normalize(k);
    const cosTh = dot(anorm, bnorm);
    const sinTh = l2norm(k);
    const res = vc * @as(Vec3, @splat(cosTh)) + cross(knorm, vc) * @as(Vec3, @splat(sinTh)) + knorm * @as(Vec3, @splat(dot(knorm, vc) * (1 - cosTh)));
    return res;
}

test "geometry. asin()" {
    print2(@src(), "\n asin(1)={} ", .{std.math.asin(@as(f32, 1.0))}); // 1.57079637 = π/2
    print2(@src(), "\n asin(-1)={}", .{std.math.asin(@as(f32, -1.0))}); // -1.57079637 = -π/2
    print2(@src(), "\n asin(0)={} ", .{std.math.asin(@as(f32, 0.0))}); // 0
    print2(@src(), "\n asin(-0)={}", .{std.math.asin(@as(f32, -0.0))}); // -0
}

pub fn testRodriguezRotation() !void {

    // check that rotating the original vector brings it to v2
    {
        var i: u32 = 0;
        while (i < 100) : (i += 1) {
            const v1 = normalize(randNormalVec3()); // start
            const v2 = normalize(randNormalVec3()); // target
            const v3 = rotateCwithRotorAB(v1, v1, v2); // try to rotate start to target
            print2(@src(), "Len After = {e:.3}\tDelta = {e:.3} \n", .{ l2norm(v3), l2norm(v2 - v3) });
        }
    }

    // check that rotating a random vector through v1^v2 doesn't change it's length, nor the dot product between two random vectors
    {
        var i: u32 = 0;
        while (i < 100) : (i += 1) {
            const a = normalize(randNormalVec3()); // new vec
            const b = normalize(randNormalVec3()); // new vec
            const v1 = normalize(randNormalVec3()); // start
            const v2 = normalize(randNormalVec3()); // target
            const dot1_ = dot(v1, v2);
            const v3 = rotateCwithRotorAB(v1, a, b); // try to rotate start to target
            const v4 = rotateCwithRotorAB(v2, a, b); // try to rotate start to target
            const dot2_ = dot(v3, v4);
            print2(@src(), "len(pre) {d:10.5} len(post) {d:10.5} dot(pre) {d:10.5} dot(post) {d:10.5} delta={e:13.5}\n", .{ l2norm(v1), l2norm(v3), dot1_, dot2_, dot2_ - dot1_ });
            try std.testing.expect(dot1_ - dot2_ < 1e-6);
        }
    }
}

test "geometry. rodriguez rotations" {
    try testRodriguezRotation();
}

// A ray describes a line which passes by a point p0.
// What is the point on the line closest to p0 ?
pub fn closestApproachRayPt(r0: Ray, p0: Vec3) Vec3 {
    const v1 = normalize(r0.pt1 - r0.pt0);
    const a = dot(v1, p0);
    const res = Vec3{ a, a, a } * v1;
    return res;
}

// find the two points defining the bounding box of a set of points in 3D
pub fn bounds3(pts: []Vec3) [2]Vec3 {
    var _min: Vec3 = pts[0];
    var _max: Vec3 = pts[0];
    for (pts[1..]) |p| {
        if (p[0] < _min[0]) _min[0] = p[0];
        if (p[1] < _min[1]) _min[1] = p[1];
        if (p[2] < _min[2]) _min[2] = p[2];
        if (p[0] > _max[0]) _max[0] = p[0];
        if (p[1] > _max[1]) _max[1] = p[1];
        if (p[2] > _max[2]) _max[2] = p[2];
    }
    return .{ _min, _max };
}

pub fn bounds2(pts: anytype) [2][2]f32 {
    var _min = pts[0];
    var _max = pts[0];
    for (pts[1..]) |p| {
        if (p[0] < _min[0]) _min[0] = p[0];
        if (p[1] < _min[1]) _min[1] = p[1];
        // if (p[2]<_min[2]) _min[2]=p[2];
        if (p[0] > _max[0]) _max[0] = p[0];
        if (p[1] > _max[1]) _max[1] = p[1];
        // if (p[2]>_max[2]) _max[2]=p[2];
    }
    return .{ _min, _max };
}

pub fn boundsBBox(pts: anytype) BBox {
    var _min = pts[0];
    var _max = pts[0];
    for (pts[1..]) |p| {
        if (p[0] < _min[0]) _min[0] = p[0];
        if (p[1] < _min[1]) _min[1] = p[1];
        // if (p[2]<_min[2]) _min[2]=p[2];
        if (p[0] > _max[0]) _max[0] = p[0];
        if (p[1] > _max[1]) _max[1] = p[1];
        // if (p[2]>_max[2]) _max[2]=p[2];
    }
    // return .{_min,_max};
    return .{ .x = .{ .hi = _max[0], .lo = _min[0] }, .y = .{ .hi = _max[1], .lo = _min[1] } };
}

pub fn clipf32(a: f32, mina: f32, maxa: f32) f32 {
    if (a == std.math.inf(f32))
        return maxa;
    if (a == -std.math.inf(f32))
        return mina;
    if (a == std.math.nan(f32))
        return maxa;
    if (a == -std.math.nan(f32))
        return mina;
    if (a > maxa)
        return maxa;
    if (a < mina)
        return mina;
    return a;
}

pub fn clipi(comptime T: type, a: T, mina: T, maxa: T) T {
    if (a > maxa) return maxa;
    if (a < mina) return mina;
    return a;
}
