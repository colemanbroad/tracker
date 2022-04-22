usingnamespace ("geometry.zig");

const draw = @import("drawing.zig");
const drawPoints3DMovie = draw.drawPoints3DMovie;
const drawMesh3DMovie2 = draw.drawMesh3DMovie2;

const home = "/Users/broaddus/Desktop/work/zig-tracker/test-artifacts/";

test "geometry. mesh. subdivideMesh()" {
    // pub fn main() !void {
    const home = test_home ++ "polysurf3D/";
    // try mkdirIgnoreExistsAbsolute(home);

    var box = BoxPoly.createAABB(.{ -3, -3, -3 }, .{ 5, 3, 4 });
    // box.vs[0] = .{1,2,3};
    // @compileLog(@TypeOf(box.vs[0..]));
    // const surf1 = Mesh{.vs=box.vs[0..] , .es=box.es[0..] , .fs=box.fs[0..]};
    // const surf1 = box.toMesh();
    var a = box.vs.len;
    var b = box.es.len;
    var c = box.fs.len; // TODO: FIXME: I shouldn't have to do this just to convert types....
    const surf1 = Mesh{ .vs = box.vs[0..a], .es = box.es[0..b], .fs = box.fs[0..c] };

    // const surf = try subdivideCurve(surf1 , 5);
    // try drawMesh3DMovie2(surf,"polysurf3D/img");

    const surf2 = try subdivideMesh(surf1, 5);
    defer surf2.deinit();
    try drawMesh3DMovie2(surf2, home ++ "img");
}

test "geometry. mesh. Chaikin Curve" {
    const npts: u32 = 10;
    // var pts = try allocator.alloc(Vec3,npts*(1<<nsubdiv));
    const pts0 = try allocator.alloc(Vec3, npts);
    defer allocator.free(pts0);
    // var pts0:[npts]Vec3 = undefined;
    pts0[0] = randNormalVec3();
    const dx = Vec3{ 0.01, 0.01, 0.01 };

    var i: u32 = 1;
    while (i < npts) : (i += 1) {
        pts0[i] = pts0[i - 1] + randNormalVec3() * dx;
    }

    const curve = try chaikinPeriodic(pts0);
    defer allocator.free(curve);

    const home = test_home ++ "chaikin/";
    // try mkdirIgnoreExistsAbsolute(home);
    _ = try drawPoints3DMovie(curve, home ++ "img");
}
