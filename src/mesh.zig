/// Defines Mesh3D type. Contains vertices in R3, Edges and (triangular) Faces.
/// Defines simple drawing functions on this mesh type.
///
///
///
///
///

// test "test mesh api 0" {
//     var al = ArenaAlloc();
//     const vs = randomVertices(al,100);
//     const mesh = Mesh3D{.vs}
// }

const std = @import("std");
const im = @import("image_base.zig");
const geo = @import("geometry.zig");

const Img2D = im.Img2D;
const Vec2 = geo.Vec2;
const Vec3 = geo.Vec3;

// const sphereTrajectory = geo.sphereTrajectory;
// const bounds3 = geo.bounds3;
// const vec2 = geo.vec2;
const l2norm = geo.l2norm;
const uvec2 = geo.uvec2;

const min3 = std.math.min3;
const max3 = std.math.max3;
const print = std.debug.print;
const assert = std.debug.assert;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var allocator = gpa.allocator();

const test_home = "/Users/broaddus/work/isbi/zig-tracker/test-artifacts/mesh/";

test {
    std.testing.refAllDecls(@This());
}

pub inline fn inbounds(img: anytype, px: anytype) bool {
    if (0 <= px[0] and px[0] < img.nx and 0 <= px[1] and px[1] < img.ny) return true else return false;
}

pub fn idxsortFn(zvals: []f32, lhs: u32, rhs: u32) bool {
    if (zvals[lhs] < zvals[rhs]) return true else return false;
}

// faces are composed of 4 vertices
pub const Mesh3D = struct {
    const This = @This();

    vs: []Vec3,
    es: [][2]u32,
    fs: ?[][4]u32,

    pub fn deinit(this: This, al: std.mem.Allocator) void {
        al.free(this.vs);
        al.free(this.es);
        if (this.fs) |fs| al.free(fs);
    }
};

// render mesh on top of pic. all white. fill in lines and faces. pure zig.
// pub fn drawMesh2D(mesh: Mesh2D, pic: Img2D([4]u8)) !void {}

/// inplace sorting of faces by z-position
fn faceZOrder(vertices: []Vec3, faces: [][4]u32) ![]u32 {
    const zpos = try allocator.alloc(f32, faces.len);
    defer allocator.free(zpos);
    const indices = try allocator.alloc(u32, faces.len);
    // defer allocator.free(indices);

    for (faces, 0..) |f, i| {
        const zval = (vertices[f[0]][0] + vertices[f[1]][0] + vertices[f[2]][0] + vertices[f[3]][0]) * 0.25; //Vec3{0.25,0.25,0.25};
        zpos[i] = zval;
        indices[i] = @intCast(u32, i);
    }

    // print("inds[..10]={d}\n",.{indices[0..10].*});
    std.sort.sort(u32, indices, zpos, idxsortFn);
    // print("inds[..10]={d}\n",.{indices[0..10].*});
    return indices;
}

pub fn myabs(a: anytype) @TypeOf(a) {
    if (a < 0) return -a else return a;
}

/// Render a filled triangle. tri in X,Y coords.
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
    var right = Vec2{ @floor(xmid), b[1] };
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

test "test renderFilledQuad() and renderFilledTri()" {
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

// Generate a rectangular grid polygon
pub fn gridMesh(nx: u32, ny: u32) !Mesh3D {
    var vs = try allocator.alloc(Vec3, nx * ny);
    var es = try allocator.alloc([2]u32, 2 * nx * ny);
    var fs = try allocator.alloc([4]u32, nx * ny);

    // var i:u32=0;
    // var j:u32=0;
    var nes: u32 = 0; // n edges
    var nfs: u32 = 0; // n faces

    for (vs, 0..) |_, i| {
        const k = @intCast(u32, i);
        // coords
        const x = i % nx;
        const y = i / nx;

        // verts
        vs[i] = Vec3{ 0, @intToFloat(f32, y), @intToFloat(f32, x) };

        // edges
        if (x < nx - 1) {
            es[nes] = .{ k, k + 1 };
            nes += 1;
        }
        if (y < ny - 1) {
            es[nes] = .{ k, k + nx };
            nes += 1;
        }

        // faces
        if (x < nx - 1 and y < ny - 1) {
            fs[nfs] = .{ k, k + 1, k + nx + 1, k + nx };
            nfs += 1;
        }
    }

    es = try allocator.realloc(es, es[0..nes].len);
    fs = try allocator.realloc(fs, fs[0..nfs].len);
    return Mesh3D{ .vs = vs, .es = es, .fs = fs };
}

// const boundaryConditions = enum {
//   Periodic,
//   Constant,
// };

pub const BoxPoly = struct {
    vs: [8]Vec3,
    es: [12][2]u32,
    fs: [6][4]u32,

    /// See [Hasse Diagram][]
    pub fn createAABB(low: Vec3, hig: Vec3) BoxPoly {

        // Vertices
        var vs: [8]Vec3 = undefined;
        // 0
        vs[0] = low;
        // 1
        vs[1] = Vec3{ low[0], low[1], hig[2] };
        vs[2] = Vec3{ low[0], hig[1], low[2] };
        vs[3] = Vec3{ hig[0], low[1], low[2] };
        // 2
        vs[4] = Vec3{ hig[0], hig[1], low[2] };
        vs[5] = Vec3{ hig[0], low[1], hig[2] };
        vs[6] = Vec3{ low[0], hig[1], hig[2] };
        // 3
        vs[7] = hig;

        // Edges
        var es: [12][2]u32 = undefined;
        es[0] = .{ 0, 1 };
        es[1] = .{ 0, 2 };
        es[2] = .{ 0, 3 };
        es[3] = .{ 1, 5 };
        es[4] = .{ 1, 6 };
        es[5] = .{ 2, 4 };
        es[6] = .{ 2, 6 };
        es[7] = .{ 3, 4 };
        es[8] = .{ 3, 5 };
        es[9] = .{ 4, 7 };
        es[10] = .{ 5, 7 };
        es[11] = .{ 6, 7 };

        // Faces
        // TODO: is it normal to describe faces in terms of vertices or edges ? Vertices seems to feel right...
        // But we could adopt the convention that the order of vertices is also the order of the edges, making them nearly equivalent.
        var fs: [6][4]u32 = undefined;
        fs[0] = .{ 0, 1, 5, 3 };
        fs[1] = .{ 0, 2, 6, 1 };
        fs[2] = .{ 0, 3, 4, 2 };
        fs[3] = .{ 7, 4, 3, 5 };
        fs[4] = .{ 7, 5, 1, 6 };
        fs[5] = .{ 7, 6, 2, 4 };

        return .{ .vs = vs, .es = es, .fs = fs };
    }

    pub fn toMesh(this: BoxPoly) Mesh3D {
        return Mesh3D{ .vs = this.vs[0..], .es = this.es[0..], .fs = this.fs[0..] };
    }
};

// return a periodic Chaikin (subdivision) curve from control points `pts0`
// assumes points are connected in a loop
pub fn chaikinPeriodic(pts0: []Vec3) ![]Vec3 {
    const npts = @intCast(u32, pts0.len);
    const nsubdiv = 10;

    var pts = try allocator.alloc(Vec3, npts * (1 << nsubdiv) * 2); // holds all subdiv levels. NOTE: 1 + 1/2 + 1/4 + 1/8 ... = 2
    defer allocator.free(pts);
    for (pts0, 0..) |p, k| pts[k] = p;

    var idx_start: u32 = 0; // npts * (1<<i - 1);
    var idx_dx: u32 = npts;
    var idx_end: u32 = idx_start + idx_dx;

    const half = Vec3{ 0.5, 0.5, 0.5 };
    const quart = Vec3{ 0.25, 0.25, 0.25 };

    {
        var k: u32 = 0;
        var i: u32 = 0;
        while (i < nsubdiv) : (i += 1) {
            const a = pts[idx_start..idx_end];
            const b = pts[idx_end .. idx_end + 2 * idx_dx];

            // first pass. create midpoints.
            k = 0;
            while (k < a.len - 1) : (k += 1) {
                b[2 * k] = a[k];
                b[2 * k + 1] = (a[k] + a[k + 1]) * half;
            }

            // far right bounds.
            b[b.len - 1] = (a[0] + a[a.len - 1]) * half; // midpoint
            b[b.len - 2] = a[a.len - 1]; // start point

            // do convolution
            k = 1;
            while (k < a.len) : (k += 1) {
                const k2 = 2 * k;
                b[k2] = b[k2] * half + b[k2 - 1] * quart + b[k2 + 1] * quart;
            }

            // far left bounds
            b[0] = b[0] * half + b[b.len - 1] * quart + b[1] * quart;

            // move the bounds forward
            idx_start = idx_end;
            idx_dx *= 2;
            idx_end = idx_start + idx_dx;
        }
    }

    var ret = try allocator.alloc(@TypeOf(pts[0]), idx_end - idx_start);
    for (ret, 0..) |*v, i| v.* = pts[idx_start + i];

    return ret;
    // return pts[idx_start..idx_end];
}

// Only subdivides edges. Works even if surf.fs==null;
pub fn subdivideCurve(surf: Mesh3D, nsubdiv: u32) !Mesh3D {
    // const nsubdiv = 5;

    // Algorithm:
    // 1. Add new vertex at midpoint of each edge, splitting each edge into two.
    // 2. Update the original vertices to a weighted avg over neighbours + old position.
    // 3. Repeat.
    // We utilize an n_vertices x n_maxneibs array for fast mapping from a vertex -> neighbour list.

    // For surfaces:
    // Split each face in the middle. square faces get split into four little squares. triangles are split into 3 triangles.
    // Move each old face to a weighted average over it's neighbours
    // split every edge. add a new vertex at the midpoint. old edges are replaced with two new edges.
    // every old vertex is updated to the weighted avg of it's self + (new) neibs. This requires being able to query neibs for a given vertex.
    // we can do this by storing neib ids in an array mapping id -> [n]id . but should we store each edge once? or do we store a->b and b->a ?
    // could we use a spatial tree at all? no we need to memorize exact neibs and they may not be spatially close. and anyways a raw lookup is faster.
    // We need an efficient sparse graph datastructure. Two-way, vertex -> [n]vertex in both directions. But this is only efficient if most vertices have
    // the same number of edges (and none have more than n).

    var nvs = @intCast(u32, surf.vs.len);
    var nes = @intCast(u32, surf.es.len);
    // var nfs = @intCast(u32,surf.fs.?.len);

    const half = Vec3{ 0.5, 0.5, 0.5 };
    // const quart = Vec3{0.25,0.25,0.25};

    // assign a large amount of memory to hold all the points
    var verts = try allocator.alloc(Vec3, nvs * 100);
    var edges = try allocator.alloc([2]u32, nes * 100);
    var newedges = try allocator.alloc([2]u32, nes * 100);
    // var faces = try allocator.alloc([4]u32,nfs*100);

    // init
    for (surf.vs, 0..) |v, i| {
        verts[i] = v;
    }
    for (surf.es, 0..) |v, i| {
        edges[i] = v;
    }
    // for (surf.fs) |v,i| {faces[i] = v;}

    var subdivcount: u8 = 0;
    while (subdivcount < nsubdiv) : (subdivcount += 1) {

        // NOTE: underlying memory is updated in-place
        const oldverts = verts[0..nvs];

        // create new edges. exactly 2x number of old edges.
        for (edges[0..nes], 0..) |e, i| {
            const v0 = verts[e[0]];
            const v1 = verts[e[1]];
            const newvert = (v0 + v1) * half;
            verts[nvs] = newvert;
            newedges[2 * i] = .{ e[0], nvs };
            newedges[2 * i + 1] = .{ nvs, e[1] };
            nvs += 1;
            nes += 1; // splitting an edge only adds one edge to the total
        }

        // build VertexNeibArray structure for fast vertex neib access.
        const na = try VertexNeibArray(3).init(newedges[0..nes], nvs);

        // update position of all old vertices.
        for (oldverts, 0..) |*v, i| {
            const nn = na.count[i];
            const ns = na.neibs[i];
            if (nn > 1) { // don't update positions of vertices with only 1 neib (or zero)
                var newpos = v.* * Vec3{ 2, 2, 2 }; // self wegiht is double neib weight
                for (ns[0..nn]) |n| {
                    newpos += verts[n]; // Vec3 neib position with weight 1. NOTE: vertex may be new! i.e. not in oldverts.
                }
                newpos /= Vec3{ @intToFloat(f32, nn + 2), @intToFloat(f32, nn + 2), @intToFloat(f32, nn + 2) }; // normalize
                v.* = newpos;
            }
        }

        // swap edge list pointers
        const _tmp = newedges;
        newedges = edges;
        edges = _tmp;
    }

    _ = allocator.resize(verts, nvs);
    _ = allocator.resize(edges, nes);

    // return Mesh{.vs=verts[0..nvs] , .es=edges[0..nes] , .fs=null};
    return Mesh3D{ .vs = verts, .es = edges, .fs = null };
}

// Subdivides edges and quad faces
pub fn subdivideMesh(surf: Mesh3D, nsubdiv: u32) !Mesh3D {
    // const nsubdiv = 5;

    // Algorithm:
    // 1. Add new vertex at midpoint of each edge, splitting each edge into two.
    // 2. Update the original vertices to a weighted avg over neighbours + old position.
    // 3. Repeat.
    // We utilize an n_vertices x n_maxneibs array for fast mapping from a vertex -> neighbour list.

    // For surfaces:
    // Split each face in the middle. square faces get split into four little squares. triangles are split into 3 triangles.
    // Move each old face to a weighted average over it's neighbours
    // split every edge. add a new vertex at the midpoint. old edges are replaced with two new edges.
    // every old vertex is updated to the weighted avg of it's self + (new) neibs. This requires being able to query neibs for a given vertex.
    // we can do this by storing neib ids in an array mapping id -> [n]id . but should we store each edge once? or do we store a->b and b->a ?
    // could we use a spatial tree at all? no we need to memorize exact neibs and they may not be spatially close. and anyways a raw lookup is faster.
    // We need an efficient sparse graph datastructure. Two-way, vertex -> [n]vertex in both directions. But this is only efficient if most vertices have
    // the same number of edges (and none have more than n).

    const nvs0 = @intCast(u32, surf.vs.len);
    const nes0 = @intCast(u32, surf.es.len);
    const nfs0 = @intCast(u32, surf.fs.?.len);

    const half = Vec3{ 0.5, 0.5, 0.5 };
    const quart = Vec3{ 0.25, 0.25, 0.25 };

    // assign a large amount of memory to hold all the points
    // how much memory is necessary? We know that faces will multiply by 4x with each subdivision, and edges will double.
    // vertices are more complicated, but will increase by 1 for each edge and one for each face, which means they will more than 4x
    // with each subdivision (but with a 1-round delay).

    // const n_faces_final = nfs * @exp2(2*nsubdiv);
    // const n_edges_final = nes * @exp2(nsubdiv);
    // const n_vertices_final = nes * @exp2(nsubdiv);

    // compute the exact number of vertices, edges and faces we will have at the end.
    const n_final = blk: {
        var nv = nvs0;
        var ne = nes0;
        var nf = nfs0;
        var count: u32 = 0;
        while (count < nsubdiv) : (count += 1) {
            nv += ne + nf;
            ne *= 2;
            ne += 4 * nf;
            nf *= 4;
        }
        break :blk .{ nv, ne, nf };
    };

    var verts = try allocator.alloc(Vec3, n_final[0]);
    var n_verts: u32 = 0;
    // defer allocator.free(verts);
    var oldedges = try allocator.alloc([2]u32, n_final[1]);
    var n_oldedges: u32 = 0;
    // defer allocator.free(oldedges);
    var newedges = try allocator.alloc([2]u32, n_final[1]);
    var n_newedges: u32 = 0;
    defer allocator.free(newedges);
    var oldfaces = try allocator.alloc([4]u32, n_final[2]);
    var n_oldfaces: u32 = 0;
    // defer allocator.free(oldfaces);
    var newfaces = try allocator.alloc([4]u32, n_final[2]);
    var n_newfaces: u32 = 0;
    defer allocator.free(newfaces);

    // init
    for (surf.vs, 0..) |v, i| {
        verts[i] = v;
        n_verts += 1;
    }
    for (surf.es, 0..) |v, i| {
        oldedges[i] = v;
        n_oldedges += 1;
    }
    for (surf.fs.?, 0..) |v, i| {
        oldfaces[i] = v;
        n_oldfaces += 1;
    }

    var subdivcount: u8 = 0;
    while (subdivcount < nsubdiv) : (subdivcount += 1) {

        // NOTE: underlying memory is updated in-place
        // unlike oldedges/oldfaces no verts are deleted, just updated. so we can keep a single, growing array for verts.
        // const oldverts = verts[0..nvs];
        // const n_oldverts = verts.;
        // number of vertices, oldedges and oldfaces.
        // these update as we add v,e,f to the array.
        // var nvs = 0; //  we've added
        // var nes = 0; // updates as we add new oldedges
        // var nfs = 0; // updates as we add new oldfaces

        const n_oldverts = n_verts; // only grows

        // split each edge. create new vertex and two new oldedges.
        for (oldedges[0..n_oldedges]) |e| {
            const v0 = verts[e[0]];
            const v1 = verts[e[1]];
            const newvert1 = (v0 + v1) * half;
            verts[n_verts] = newvert1;
            n_verts += 1;
            newedges[n_newedges] = .{ e[0], n_verts - 1 };
            n_newedges += 1;
            newedges[n_newedges] = .{ n_verts - 1, e[1] };
            n_newedges += 1;
        }

        // const n_edgeverts = nes;

        // map old face idx to old edge idx (and thus to new vertices = verts[oldverts.len + edge_idx])
        // we will use this map
        const f2e = try Face2Edge(4).init(oldedges[0..n_oldedges], oldfaces[0..n_oldfaces]);
        defer f2e.deinit();

        // Create new faces and edges! each quad face adds 1 new vert and 4 new edgs that it owns completely,
        // plus 4 new verts and 4 oldedges that it shares (each with one other = avg of 2vs and 2es).
        // exactly 4x number of oldfaces after each round.
        // ASSUMES QUAD FACES!

        for (oldfaces[0..n_oldfaces], 0..) |f, i| {
            const v0 = verts[f[0]];
            const v1 = verts[f[1]];
            const v2 = verts[f[2]];
            const v3 = verts[f[3]];
            const newvert2 = (v0 + v1 + v2 + v3) * quart;
            const newvert2idx = n_verts;
            verts[n_verts] = newvert2;
            n_verts += 1;

            // get indices of new edge vertices. The nth new vertex added <= nth edge allows this formula to work.
            const newvert1idxA = f2e.neibs[i][0] + n_oldverts;
            const newvert1idxB = f2e.neibs[i][1] + n_oldverts;
            const newvert1idxC = f2e.neibs[i][2] + n_oldverts;
            const newvert1idxD = f2e.neibs[i][3] + n_oldverts;

            // add four new edges from new face vertex out to each new edge vertex
            newedges[n_newedges] = .{ newvert2idx, newvert1idxA };
            n_newedges += 1;
            newedges[n_newedges] = .{ newvert2idx, newvert1idxB };
            n_newedges += 1;
            newedges[n_newedges] = .{ newvert2idx, newvert1idxC };
            n_newedges += 1;
            newedges[n_newedges] = .{ newvert2idx, newvert1idxD };
            n_newedges += 1;

            // split each face into four new faces;
            // WARNING! The encoding assumes that newvert1idxA is inserted between v0,v1, etc.
            newfaces[n_newfaces] = .{ f[0], newvert1idxA, newvert2idx, newvert1idxD };
            n_newfaces += 1;
            newfaces[n_newfaces] = .{ f[1], newvert1idxB, newvert2idx, newvert1idxA };
            n_newfaces += 1;
            newfaces[n_newfaces] = .{ f[2], newvert1idxC, newvert2idx, newvert1idxB };
            n_newfaces += 1;
            newfaces[n_newfaces] = .{ f[3], newvert1idxD, newvert2idx, newvert1idxC };
            n_newfaces += 1;
        }

        // build VertexNeibArray structure for fast vertex neib access. Max 2 neibs / vertex.
        const na = try VertexNeibArray(4).init(newedges[0..n_newedges], n_verts);
        defer na.deinit();

        // update position of all old vertices.
        for (verts[0..n_oldverts], 0..) |*v, i| {
            const nn = na.count[i];
            const ns = na.neibs[i];
            if (nn > 1) { // don't update positions of vertices with only 1 neib (or zero)
                var newpos = v.* * Vec3{ 4, 4, 4 }; // self wegiht is 4x neib weight
                for (ns[0..nn]) |n| {
                    newpos += verts[n]; // Vec3 neib position with weight 1. NOTE: vertex may be new! i.e. not in oldverts.
                }
                newpos /= Vec3{ @intToFloat(f32, nn + 4), @intToFloat(f32, nn + 4), @intToFloat(f32, nn + 4) }; // normalize
                v.* = newpos;
            }
        }

        // swap edge list pointers
        const _tmp1 = newedges;
        newedges = oldedges;
        oldedges = _tmp1;

        // swap edge list pointers
        const _tmp2 = n_newedges;
        n_newedges = n_oldedges;
        n_oldedges = _tmp2;

        // swap edge list pointers
        const _tmp3 = newfaces;
        newfaces = oldfaces;
        oldfaces = _tmp3;

        // swap edge list pointers
        const _tmp4 = n_newfaces;
        n_newfaces = n_oldfaces;
        n_oldfaces = _tmp4;

        n_newedges = 0;
        n_newfaces = 0;
    }

    _ = allocator.resize(verts, n_verts);
    _ = allocator.resize(oldedges, n_oldedges);
    _ = allocator.resize(oldfaces, n_oldfaces);

    // return Mesh{.vs=verts[0..n_verts] , .es=oldedges[0..n_oldedges] , .fs=oldfaces[0..n_oldfaces]};
    return Mesh3D{ .vs = verts, .es = oldedges, .fs = oldfaces };
}

// index -> [n]index map. Maps vertices to their neighbours.
pub fn VertexNeibArray(comptime nneibs: u8) type {
    return struct {
        const Self = @This();
        count: []u8,
        neibs: [][nneibs]u32,

        // convert Edgelist into vertex neighbour array
        pub fn init(es: [][2]u32, nvert: usize) !Self {
            const empty: u32 = ~@as(u32, 0); // maximum u32 val

            var neibCount = try allocator.alloc(u8, nvert);
            for (neibCount) |*v| v.* = 0;
            var neibs = try allocator.alloc([nneibs]u32, nvert);
            for (neibs) |*v| v.* = [1]u32{empty} ** nneibs;

            // for each edge in the input edgelist add two entries to the VertexNeibArray:
            // NOTE: this assumes the edgelist only store edges once (only a->b not also b->a)
            for (es) |e| {
                const e0 = e[0];
                const e1 = e[1];
                neibs[e0][neibCount[e0]] = e1;
                neibCount[e0] += 1;
                neibs[e1][neibCount[e1]] = e0;
                neibCount[e1] += 1;
            }

            return Self{ .count = neibCount, .neibs = neibs };
        }

        pub fn deinit(self: Self) void {
            allocator.free(self.count);
            allocator.free(self.neibs);
        }
    };
}

test "geometry. mesh. VertexNeibArray on BoxPoly" {
    var box = BoxPoly.createAABB(.{ 0, 0, 0 }, .{ 1, 1, 1 });
    const surf = Mesh3D{ .vs = box.vs[0..], .es = box.es[0..], .fs = box.fs[0..] };
    const nl = try VertexNeibArray(3).init(surf.es, surf.vs.len);
    defer nl.deinit();
    print("\n{any}\n", .{nl});
}

// index -> [n]index map. Maps edges to their faces and faces to their edges.
pub fn Face2Edge(comptime nneibs: u8) type {
    return struct {
        const Self = @This();
        count: []u8,
        neibs: [][nneibs]u32,

        // pub fn init(es:[][2]u32 , fs:[][4]u32 , nfaces:u32) !Self {
        pub fn init(
            es: [][2]u32,
            fs: [][4]u32,
        ) !Self {
            // const empty:u32 = ~@as(u32,0); // maximum u32 val

            var face2edge = try allocator.alloc([nneibs]u32, fs.len);
            var count = try allocator.alloc(u8, fs.len);

            var map = std.AutoHashMap([2]u32, u32).init(allocator);
            defer map.deinit();
            for (es, 0..) |e, i| {
                try map.put(e, @intCast(u32, i));
                try map.put(.{ e[1], e[0] }, @intCast(u32, i));
            } // Add fwd and backward edges to map. ASSUME: Edges are undirected.
            for (fs, 0..) |f, i| {
                for (f[0 .. f.len - 1], 0..) |_, j| { // iterate over all verts but last. ASSUME: faces are closed polygons (embedded in 3D).
                    face2edge[i][j] = map.get(.{ f[j], f[j + 1] }).?;
                }
                face2edge[i][f.len - 1] = map.get(.{ f[f.len - 1], f[0] }).?; // attatch last vertex to first
                count[i] = f.len;
            }
            return Self{ .count = count, .neibs = face2edge };
        }

        pub fn deinit(self: Self) void {
            allocator.free(self.count);
            allocator.free(self.neibs);
        }
    };
}

test "geometry. mesh. Face2Edge on BoxPoly" {
    var box = BoxPoly.createAABB(.{ 0, 0, 0 }, .{ 1, 1, 1 });
    const surf = Mesh3D{ .vs = box.vs[0..], .es = box.es[0..], .fs = box.fs[0..] };
    const e2f = try Face2Edge(4).init(surf.es, surf.fs.?);
    defer e2f.deinit();
    print("\n{any}\n", .{e2f});
}
