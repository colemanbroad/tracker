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

const Vector = std.meta.Vector;
pub const Vec3 = Vector(3, f32);

test {
    std.testing.refAllDecls(@This());
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
    for (phis) |*v, i| v.* = ((@intToFloat(f32, i)) / 99) * pi;
    var thetas: [100]f32 = undefined;
    // for (thetas) |*v,i| v.* = ((@intToFloat(f32,i)+1)/105) * 2*pi;
    for (thetas) |*v, i| v.* = ((@intToFloat(f32, i)) / 99) * 2 * pi;
    var pts: [100]Vec3 = undefined;
    for (pts) |*v, i| v.* = Vec3{ @cos(phis[i]), @sin(thetas[i]) * @sin(phis[i]), @cos(thetas[i]) * @sin(phis[i]) }; // ZYX coords
    return pts;
}

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

    pub fn toMesh(this: BoxPoly) Mesh {
        return Mesh{ .vs = this.vs[0..], .es = this.es[0..], .fs = this.fs[0..] };
    }
};

// faces are composed of 4 vertices
pub const Mesh = struct {
    const This = @This();
    vs: []Vec3,
    es: [][2]u32,
    fs: ?[][4]u32,

    pub fn deinit(this: This) void {
        allocator.free(this.vs);
        allocator.free(this.es);
        if (this.fs) |fs| allocator.free(fs);
    }
};

const Allocator = std.mem.Allocator;

// A Mesh object consisting of triangles with vertices embedded in 2D
// Edges are implied by Triangles, i.e. they can't exist by themselves.
// ArrayList(Pt) is necessary because:
//  - can shift point positions while maintaining mesh
//  - multiple points potentially at same position (temporarily)
//  - test idx for equality instead of float values
//  - save mem in Tri{} objects as one point could be in 
//  - BUT this makes removal harder... to remove a point we must either
//     - remove from self.vs and shift so self.vs is compact
//     - replace in self.vs with null or some other special value
//     - ignore it in self.vs. a point is effectively removed if no Tri or Edge point to it.
// Q: removing a point should remove associated edges and triangles ?
// Q: removing a triangle doesn't automatically remove points, because triangles share points.

// All of the relationships are many-many. point → [2,n] edges, edge → 2 pts, point → [1,n] tris, tri → 3 pts, edge → [1,2] tris (in 2d), tri → 3 edges.
// This makes it tricky to define ownership rules.

// For Bower-Watson Alg we only need triangles. We want to be able to traverse triangles quickly, and add and remove them.

pub const Mesh2D = struct {

    const Self = @This();

    const Pt = Vec2;
    const PtIdx = u32;
    const Edge = [2]PtIdx;
    const Tri  = [3]PtIdx;
    const TriIdx = u32;

    // const Tri = struct {
    //     tri:[3]PtIdx,
    //     // idx:TriIdx,
    //     neibs:[3]?TriIdx,
    // };

    vs: std.ArrayList(Pt),
    // es: std.ArrayList(Edge),
    // ts: std.ArrayList(?Tri),
    ts: std.AutoHashMap(Tri, void),

    // edgeset: std.AutoHashMap(Edge, void),
    edge2tri: std.AutoHashMap(Edge, [2]?Tri), // can also keep track of edge existence

    al: Allocator,


    // pub const Iterator = struct {
    //     hm: *const Self,
    //     index: u32 = 0,

    //     pub fn next(it: *Iterator) ?*Tri {
    //         assert(it.index <= it.hm.capacity());
    //         if (it.hm.size == 0) return null;

    //         const cap = it.hm.capacity();
    //         const end = it.hm.metadata.? + cap;
    //         var metadata = it.hm.metadata.? + it.index;

    //         while (metadata != end) : ({
    //             metadata += 1;
    //             it.index += 1;
    //         }) {
    //             if (metadata[0].isUsed()) {
    //                 const key = &it.hm.keys()[it.index];
    //                 const value = &it.hm.values()[it.index];
    //                 it.index += 1;
    //                 return Entry{ .key_ptr = key, .value_ptr = value };
    //             }
    //         }

    //         return null;
    //     }
    // };

    // tri_neibs: [][3]?TriIdx,
    // pt_neibs : [][14]?PtIdx,

    pub fn deinit(self: *Self) void {
        self.vs.deinit();
        // self.es.deinit();
        self.ts.deinit();
        self.edge2tri.deinit();
    }

    pub fn init(a:Allocator) !Self {
        var s = Self{
                .al = a ,
                .vs = try std.ArrayList(Pt).initCapacity(a, 100)   ,
                .ts = std.AutoHashMap(Tri, void).init(a),
                .edge2tri = std.AutoHashMap(Edge, [2]?Tri).init(a) ,
            };
        return s;
    }

    pub fn initRand(a:Allocator) !Self {

        var s = Self{
                    .al = a ,
                    .vs = try std.ArrayList(Pt).initCapacity(a, 100)   ,
                    .ts = std.AutoHashMap(Tri, void).init(a),
                    .edge2tri = std.AutoHashMap(Edge, [2]?Tri).init(a) ,
                };

        // update pts
        {var i:u8=0; while (i<4):(i+=1) {
            const fi = @intToFloat(f32, i);
            const x = 6*@mod(fi,2.0) + random.float(f32);
            const y = 6*@floor(fi/2.0) + random.float(f32);
            s.vs.appendAssumeCapacity(.{x,y});
        }}

        // update tri's
        try s.ts.put(Tri{0,1,2} , {});
        try s.ts.put(Tri{1,2,3} , {});

        // update edge→tri map
        try s.addTrisToEdgeMap(try s.validTris(a));

        return s;
    }


    pub fn validTris(self:Self , a:Allocator) ![]Tri {
        const tris = blk: {
            var tri_no_null = try a.alloc(Mesh2D.Tri, self.vs.items.len);
            var tail:u8=0;
            var it = self.ts.keyIterator();
            while (it.next()) |tri| {
                tri_no_null[tail] = tri.*; 
                tail+=1;
            }
            break :blk a.resize(tri_no_null, tail).?;
        };
        return tris;
    }


    // add point return it's index in arraylist
    pub fn addPt(self:*Self , pt:Pt) !PtIdx {
        try self.vs.append(pt);
        return @intCast(PtIdx, self.vs.items.len - 1 );
    }

    // add triangles whose points are already there
    pub fn addTris(self:*Self , tris:[]Tri) !void {
        for (tris) |tri| {
            const tri_canonical = sortTri(tri);
            assert(tri_canonical[2] < self.vs.items.len); // make sure tri is valid
            try self.ts.put(tri,{});
        }
        try self.addTrisToEdgeMap(tris);
    }

    // remove triangles. remove edges iff no triangle exists.
    pub fn removeTris(self:*Self , tris:[]Tri) !void {
        try self.removeTrisFromEdgeMap(tris);
        // TODO: what good is self.ts if we don't remove bad triangles ?
        for (tris) |tri| {
            _ = self.ts.remove(tri);
            // tri.* = null;
        }
        // TODO: also, what good are self.es ?
    }


    pub fn addTrisToEdgeMap(self:*Self, tris:[]Tri) !void {
        for (tris) |tri| {
            const tri_canonical = sortTri(tri);
            const a = tri_canonical[0];
            const b = tri_canonical[1];
            const c = tri_canonical[2];
            try self.mapEdgeToTri(.{a,b}, tri_canonical);
            try self.mapEdgeToTri(.{b,c}, tri_canonical);
            try self.mapEdgeToTri(.{a,c}, tri_canonical);
        }
    }

    pub fn removeTrisFromEdgeMap(self:*Self, tris:[]Tri) !void {
        for (tris) |tri| {
            const tri_canonical = sortTri(tri);
            const a = tri_canonical[0];
            const b = tri_canonical[1];
            const c = tri_canonical[2];
            try self.remTriFromEdge(.{a,b}, tri_canonical);
            try self.remTriFromEdge(.{b,c}, tri_canonical);
            try self.remTriFromEdge(.{a,c}, tri_canonical);
        }
    }


    const eql = std.mem.eql;

    pub fn mapEdgeToTri(self:*Self, _e:Edge, tri:Tri) !void {
        const e = sortEdge(_e);
        var _entry = self.edge2tri.getPtr(e);
        if (_entry==null) {try self.edge2tri.put(e, .{tri,null}); return;}
        var entry = _entry.?;

        if (entry[0]==null) return error.InconsistentEdgeState;

        // at least one entry must be non-null
        if (eql(u32, &entry[0].? , &tri)) return; // already exists
        if (entry[1]==null) {entry[1] = tri; return;} // add it
        if (eql(u32, &entry[1].? , &tri)) return; // already exists

        // map was already full. we're trying to add a tri without deleting existing ones first.
        return error.EdgeMapFull;
    }

    // we know e maps to tri already. if it's not there, throw err.
    pub fn remTriFromEdge(self:*Self, _e:Edge, tri:Tri) !void {
        const e = sortEdge(_e);
        var _entry = self.edge2tri.getPtr(e);
        if (_entry==null) return error.EdgeDoesntExist; // must already exist.

        var entry = _entry.?;

        // first entry can never be null
        // if (entry[0]==null) return error.InconsistentEdgeState;

        // error if null
        const v0 = entry[0].?; 
        // not null, so try matching to tri
        const b0 = eql(u32,&v0,&tri);
        // assign value to v1 if not null, otherwise test b0 and either succeed or err.
        const v1 = if (entry[1]) |ent1| ent1 else {
            // v1 is null. if tri==v0 then remove it and return.
            if (b0) {_ = self.edge2tri.remove(e); return;}
            // otherwise failure. v0!=tri and v1==null... inconsistent edge state.
            else {return error.InconsistentEdgeState;}
        };

        // v1 is not null. so test it vs tri
        const b1 = eql(u32,&v1,&tri);
        // now analyze all four cases.

        // const BB = [2]bool;

        if ( b0 and  b1) {return error.InconsistentEdgeState;}
        if ( b0 and !b1) {entry.* = .{v1,null}; return;} // shift left
        if (!b0 and  b1) {entry.* = .{v0,null}; return;} // remove 2nd position
        if (!b0 and !b1) {return error.InconsistentEdgeState;}
    }

    //  ensure it exists
    //  give it a unique label?
    //  update map from edge→tri ?
    // pub fn addTri() {} 

    // add triangles (a,b,c) for each edge (a,b) of polygon connected to centerpoint (c).
    // should we also remove any existing triangles (a,b,x) ?
    // 
    pub fn addPointInPolygon(self:*Self, pt:Pt, polygon:[]PtIdx) !void {
        for (polygon) |_,i| {
            const edge = sortEdge( Edge{polygon[i],polygon[(i+1) % polygon.len]} );
            if (self.edge2tri.get(edge)==null) return error.InvalidPolygon;
        }

        // ok now we're committed . add pt to vs, then add new edges and remove old edges.
        // try self.vs.append(pt);
        const idx = try self.addPt(pt);
        // const idx = @intCast(u32, self.vs.items.len);

        // make list of triangles to add
        var tri_list = try self.al.alloc(Tri, polygon.len);
        defer self.al.free(tri_list);
        for (polygon) |_,i| {
            const edge = Edge{polygon[i],polygon[(i+1) % polygon.len]};
            tri_list[i] = Tri{edge[0],edge[1],idx};
        }

        // now add them to mesh and update self
        try self.addTris(tri_list);
    }

    pub fn sortTri(_tri:Tri) Tri {
        var tri = _tri;
        if (tri[0]>tri[1]) swap(u32,&tri[0],&tri[1]);
        if (tri[1]>tri[2]) swap(u32,&tri[1],&tri[2]);
        if (tri[0]>tri[1]) swap(u32,&tri[0],&tri[1]);
        return tri;
    }

    pub fn sortEdge(edge:Edge) Edge {
        var e = edge;
        if (e[0]>e[1]) swap(PtIdx,&e[0],&e[1]);
        return e;
    }

    pub fn show(self:Self) void {
        print("Triangles \n",.{});

        var it_ts = self.ts.keyIterator();
        while (it_ts.next()) |tri| {
            print("{d} \n",.{tri.*});
        }

        print("Edge map \n", .{});
        var it_es = self.edge2tri.iterator();
        while (it_es.next()) |kv| {
            print("{d} → {d} \n",.{kv.key_ptr.* , kv.value_ptr.*});
        }
    }

    // up to 3 neibs ? any of them can be null;
    pub fn getTriNeibs(self:Self, tri:Tri) [3]?Tri {

        const tri_canonical = sortTri(tri);
        const a = tri_canonical[0];
        const b = tri_canonical[1];
        const c = tri_canonical[2];
        var res:[3]?Tri = undefined;
        res[0] = self.getSingleNeib(.{a,b});
        res[1] = self.getSingleNeib(.{b,c});
        res[2] = self.getSingleNeib(.{a,c});
        return res;
    }

    pub fn getSingleNeib(self:Self, tri:Tri, edge:Edge) ?Tri {
        const m2tris = self.edge2tri.get(edge);
        if (eql(u32,&tri,&m2tris[0].?)) return m2tris[1];
        return m2tris[0];
    }
    

    // pub fn walk(self:Self, start:Tri) 

    // pub fn remTri () {}

    // pub fn getTrisFromEdge(e:Edge) [2]?Tri {}

    // pub fn triExistsQ(tri:Tri) bool {}

    // pub fn walkFaces(start:Tri) []Tri {}

    // pub fn walkEdges(start:Edge) ?

};

const Pix = @Vector(2,u31);
pub fn pt2PixCast(p:Vec2) Pix {
    return Pix{@floatToInt(u31, p[0]) , @floatToInt(u31, p[1])};
}

pub fn newBBox(x0:f32,x1:f32,y0:f32,y1:f32) BBox {
    return BBox{.x=.{.lo=x0,.hi=x1},.y=.{.lo=y0,.hi=y1}};
}

pub fn affine(_p:Vec2,bb0:BBox,bb1:BBox) Vec2 {
    var p = _p;

    p -= Vec2{bb0.x.lo,bb0.y.lo};
    p *= Vec2{(bb1.x.hi-bb1.x.lo)/(bb0.x.hi-bb0.x.lo) , (bb1.y.hi-bb1.y.lo)/(bb0.y.hi-bb0.y.lo)};
    p += Vec2{bb1.x.lo,bb1.y.lo};

    return p;
}

pub fn swap(comptime T: type, a:*T , b:*T) void {
    const temp = a.*;
    a.* = b.*;
    b.* = temp;
}


pub const Range = struct { hi: f32, lo: f32 };
pub const BBox = struct { x: Range, y: Range };

// Generate a rectangular grid polygon
pub fn gridMesh(nx: u32, ny: u32) !Mesh {
    var vs = try allocator.alloc(Vec3, nx * ny);
    var es = try allocator.alloc([2]u32, 2 * nx * ny);
    var fs = try allocator.alloc([4]u32, nx * ny);

    // var i:u32=0;
    // var j:u32=0;
    var nes: u32 = 0; // n edges
    var nfs: u32 = 0; // n faces

    for (vs) |_, i| {
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
    return Mesh{ .vs = vs, .es = es, .fs = fs };
}

// const boundaryConditions = enum {
//   Periodic,
//   Constant,
// };

// return a periodic Chaikin (subdivision) curve from control points `pts0`
// assumes points are connected in a loop
pub fn chaikinPeriodic(pts0: []Vec3) ![]Vec3 {
    const npts = @intCast(u32, pts0.len);
    const nsubdiv = 10;

    var pts = try allocator.alloc(Vec3, npts * (1 << nsubdiv) * 2); // holds all subdiv levels. NOTE: 1 + 1/2 + 1/4 + 1/8 ... = 2
    defer allocator.free(pts);
    for (pts0) |p, k| pts[k] = p;

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
    for (ret) |*v, i| v.* = pts[idx_start + i];

    return ret;
    // return pts[idx_start..idx_end];
}

// Only subdivides edges. Works even if surf.fs==null;
pub fn subdivideCurve(surf: Mesh, nsubdiv: u32) !Mesh {
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
    for (surf.vs) |v, i| {
        verts[i] = v;
    }
    for (surf.es) |v, i| {
        edges[i] = v;
    }
    // for (surf.fs) |v,i| {faces[i] = v;}

    var subdivcount: u8 = 0;
    while (subdivcount < nsubdiv) : (subdivcount += 1) {

        // NOTE: underlying memory is updated in-place
        const oldverts = verts[0..nvs];

        // create new edges. exactly 2x number of old edges.
        for (edges[0..nes]) |e, i| {
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
        for (oldverts) |*v, i| {
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

    verts = allocator.shrink(verts, nvs);
    edges = allocator.shrink(edges, nes);

    // return Mesh{.vs=verts[0..nvs] , .es=edges[0..nes] , .fs=null};
    return Mesh{ .vs = verts, .es = edges, .fs = null };
}

// Subdivides edges and quad faces
pub fn subdivideMesh(surf: Mesh, nsubdiv: u32) !Mesh {
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
    for (surf.vs) |v, i| {
        verts[i] = v;
        n_verts += 1;
    }
    for (surf.es) |v, i| {
        oldedges[i] = v;
        n_oldedges += 1;
    }
    for (surf.fs.?) |v, i| {
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

        for (oldfaces[0..n_oldfaces]) |f, i| {
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
        for (verts[0..n_oldverts]) |*v, i| {
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

    verts = allocator.shrink(verts, n_verts);
    oldedges = allocator.shrink(oldedges, n_oldedges);
    oldfaces = allocator.shrink(oldfaces, n_oldfaces);

    // return Mesh{.vs=verts[0..n_verts] , .es=oldedges[0..n_oldedges] , .fs=oldfaces[0..n_oldfaces]};
    return Mesh{ .vs = verts, .es = oldedges, .fs = oldfaces };
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
    const surf = Mesh{ .vs = box.vs[0..], .es = box.es[0..], .fs = box.fs[0..] };
    const nl = try VertexNeibArray(3).init(surf.es, surf.vs.len);
    defer nl.deinit();
    print("\n{d}\n", .{nl});
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
            for (es) |e, i| {
                try map.put(e, @intCast(u32, i));
                try map.put(.{ e[1], e[0] }, @intCast(u32, i));
            } // Add fwd and backward edges to map. ASSUME: Edges are undirected.
            for (fs) |f, i| {
                for (f[0 .. f.len - 1]) |_, j| { // iterate over all verts but last. ASSUME: faces are closed polygons (embedded in 3D).
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
    const surf = Mesh{ .vs = box.vs[0..], .es = box.es[0..], .fs = box.fs[0..] };
    const e2f = try Face2Edge(4).init(surf.es, surf.fs.?);
    defer e2f.deinit();
    print("\n{d}\n", .{e2f});
}

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
    print("\n\n{d}\n\n", .{invert3x3(mA)});
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
    const ipZNear = rp + dr * @splat(3, alphasNear[0]);
    const ipYNear = rp + dr * @splat(3, alphasNear[1]);
    const ipXNear = rp + dr * @splat(3, alphasNear[2]);
    const ipZFar = rp + dr * @splat(3, alphasFar[0]);
    const ipYFar = rp + dr * @splat(3, alphasFar[1]);
    const ipXFar = rp + dr * @splat(3, alphasFar[2]);

    // Test if each of the six intersection points lies inside the rectangular box face
    const p0 = box.pt0;
    const p1 = box.pt1;
    const ipZNearTest = p0[1] <= ipZNear[1] and ipZNear[1] < p1[1] and p0[2] <= ipZNear[2] and ipZNear[2] < p1[2];
    const ipYNearTest = p0[0] <= ipYNear[0] and ipYNear[0] < p1[0] and p0[2] <= ipYNear[2] and ipYNear[2] < p1[2];
    const ipXNearTest = p0[0] <= ipXNear[0] and ipXNear[0] < p1[0] and p0[1] <= ipXNear[1] and ipXNear[1] < p1[1];
    const ipZFarTest = p0[1] <= ipZFar[1] and ipZFar[1] < p1[1] and p0[2] <= ipZFar[2] and ipZFar[2] < p1[2];
    const ipYFarTest = p0[0] <= ipYFar[0] and ipYFar[0] < p1[0] and p0[2] <= ipYFar[2] and ipYFar[2] < p1[2];
    const ipXFarTest = p0[0] <= ipXFar[0] and ipXFar[0] < p1[0] and p0[1] <= ipXFar[1] and ipXFar[1] < p1[1];

    // print("Our box test results are:\n\n",.{});
    // print("pt1= {d} intersection?: {}\n",.{ipZNear, ipZNearTest});
    // print("pt2= {d} intersection?: {}\n",.{ipYNear, ipYNearTest});
    // print("pt3= {d} intersection?: {}\n",.{ipXNear, ipXNearTest});
    // print("pt4= {d} intersection?: {}\n",.{ipZFar, ipZFarTest});
    // print("pt5= {d} intersection?: {}\n",.{ipYFar, ipYFarTest});
    // print("pt6= {d} intersection?: {}\n",.{ipXFar, ipXFarTest});

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
                // print("starting pIn,pOut = {d:0.2} , {d:0.2}\n" , .{pIn,pOut});
                const temp = pIn;
                pIn = pOut;
                pOut = temp;
                // print("swapped  pIn,pOut = {d:0.2} , {d:0.2}\n" , .{pIn,pOut});
            }
        }
    }

    return .{ .pt0 = pIn, .pt1 = pOut };

    // if (ipZNearTest or ipYNearTest or ipXNearTest or ipZFarTest  or ipYFarTest  or ipXFarTest) {return Ray{.pt0=pIn , .pt1=pOut};}
    // else {return null;}
}

test "geometry. Axis aligned bounding box intersection test" {
    // fn testAxisAlignedBoundingBox() !void {
    print("\n", .{});
    {
        // The ray intersects the box at the points {0,0,0} and {1,3,1}
        const ray = Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = .{ 1, 3, 1 } };
        const box = Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = .{ 3, 3, 3 } };
        const res = intersectRayAABB(ray, box);
        try expect(abs(res.pt0.? - Vec3{ 0, 0, 0 }) < 1e-6);
        try expect(abs(res.pt1.? - Vec3{ 1, 3, 1 }) < 1e-6);
        print("{d}\n", .{res});
        print("\n", .{});
    }

    {
        // The ray doesn't intersect the box at all
        const ray = Ray{ .pt0 = .{ -1, 0, 0 }, .pt1 = .{ -1, 3, 1 } };
        const box = Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = .{ 3, 3, 3 } };
        const res = intersectRayAABB(ray, box);
        print("{d}\n", .{res});
        try expect(res.pt0 == null and res.pt1 == null);
        print("\n", .{});
    }

    {
        // The ray intersects the box at a single point {0,0,0}
        const ray = Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = .{ -1, 3, 1 } };
        const box = Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = .{ 3, 3, 3 } };
        const res = intersectRayAABB(ray, box);
        try expect((res.pt0 == null) != (res.pt1 == null)); // `!=` can be used as `xor`
        print("{d}\n", .{res});
        print("\n", .{});
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

    const intersectionPoint1 = @splat(3, abc[0]) * dr + r0;
    const intersectionPoint2 = @splat(3, abc[1]) * dp1 + @splat(3, abc[2]) * dp2 + p0;

    print("\n\nabc={d}\n\n", .{abc});
    print("intersection point (v1) = a*dr + r0 = {d} \n", .{intersectionPoint1});
    print("intersection point (v2) = b*dp1 + c*dp2 + p0 = {d} \n", .{intersectionPoint2});

    return .{ .b = abc[1], .c = abc[2], .pt = intersectionPoint2 };
}

test "geometry. Intersect Ray with (arbitrary) Face" {
    const r1 = Ray{ .pt0 = .{ 0, 0, 0 }, .pt1 = .{ 1, 1.5, 1.5 } };
    const f1 = Face{ .pt0 = .{ 5, 0, 0 }, .pt1 = .{ 0, 5, 0 }, .pt2 = .{ 0, 0, 5 } };
    const x0 = intersectRayFace(r1, f1);
    print("\n{d}\n", .{x0});
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
    const xZ = rp + dv * @splat(3, alphas[0]);
    const xY = rp + dv * @splat(3, alphas[1]);
    const xX = rp + dv * @splat(3, alphas[2]);

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
    const mindist = std.math.min3(abs2(_a), abs2(_b), abs2(_c)) * 0.1;
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

    // print("CircumCircle\n",.{});
    // print("a={d}\n",.{a});
    // print("b={d}\n",.{b});
    // print("c={d}\n",.{c});
    // print("center={d}\n",.{center});
    // print("radiusSquared={d}\n\n\n",.{radiusSquared});

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
    // print("centers:\n",.{});
    // print("0:     {d}\n",.{circumcenter});
    // print("1:     {d}\n",.{circumcenter2});
    // print("delta: {d}\n",.{circumcenter2 - circumcenter});

    // intersections
    // const p_ab = intersection(tri[0]+d01 , tri[0]+d01+rot90_d01 , tri[1]+d12 , tri[1]+d12+rot90_d12);
    const r0 = tri[0] - circumcenter;
    // const r1 = tri[1]-circumcenter;
    // const r2 = tri[2]-circumcenter;
    // print("Radii: {d} .. {d} .. {d} \n" , .{r0,r1,r2});
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
    const u = circumcircle.pt;
    const r2 = circumcircle.r2;
    const delta = pt - u;
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

    print("{b}\n", .{pointInTriangleCircumcircle2d(v1, tri)});
    print("{b}\n", .{pointInTriangleCircumcircle2d(v2, tri)});
    print("{b}\n", .{pointInTriangleCircumcircle2d(v3, tri)});
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
        print("count {d}  \n", .{count}); // , p0,p1,p2,p3});

        errdefer print("{d}\n{d}\n{d}\n{d}\n", .{ p0, p1, p2, p3 });
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
        print("count {d}  \n", .{count}); // , p0,p1,p2,p3});
        errdefer print("{d}\n{d}\n{d}\n{d}\n", .{ p0, p1, p2, p3 });
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

    print("{b}\n", .{pointInTriangle2d(v1, tri)});
    print("{b}\n", .{pointInTriangle2d(v2, tri)});
}

pub fn lerpVec2(t: f32, a: Vec2, b: Vec2) Vec2 {
    return a + Vec2{ t, t } * (b - a);
}

// VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2
// VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2
// VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2 ⚡️ VEC2 AND RAY2

pub const Ray2 = [2]Vec2;
pub const Vec2 = Vector(2, f32);

pub fn vec2(a: [2]f32) Vec2 {
    return Vec2{ a[0], a[1] };
}
pub fn uvec2(a: [2]u32) Vec2 {
    return Vec2{ @intToFloat(f32, a[0]), @intToFloat(f32, a[1]) };
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
        print("\n{d:.3}", .{cross(a, b)});
    }

    {
        const a = Vec3{ 0, 1, 0 };
        const b = Vec3{ 0, 0, 1 };
        print("\n{d:.3}", .{cross(a, b)});
    }

    {
        const a = Vec3{ 0, 1, 0 };
        const b = Vec3{ 0, 0, 1 };
        print("\n{d:.3}", .{cross(a, b)});
    }
}

pub fn dot(a: Vec3, b: Vec3) f32 {
    // return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] ;
    return @reduce(.Add, a * b);
}

pub fn abs(a: Vec3) f32 {
    return @sqrt(dot(a, a));
}

pub fn normalize(a: Vec3) Vec3 {
    const s = abs(a);
    return a / @splat(3, s);
}

pub fn normalize2(a: Vec2) Vec2 {
    const s = abs2(a);
    return a / @splat(2, s);
}

pub fn scale(a: Vec3, b: f32) Vec3 {
    return a * @splat(3, b);
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

    print("\n matVecMul : {d} ", .{matVecMul(A1, a)}); // { 0.5011608600616455, 0.7376041412353516, 0.7134442329406738 }
    // From Julia (Float64)
    // 0.5011606467452332
    // 0.7376040111844648
    // 0.7134446669414377

    print("\n matMatMul : {d} ", .{matMatMul(A1, A2)}); // { 0.1711246371269226, 0.4322627782821655, 0.6613966822624207, 0.04376308247447014, 0.3384329378604889, 0.4224681854248047, 0.9540175199508667, 1.151763677597046, 1.3093112707138062 }

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
        print("Len {} .. ", .{abs(rv)});
        const ypr = randNormalVec3() * Vec3{ 2 * pi, 2 * pi, 2 * pi };
        const rot = rotYawPitchRoll(ypr[0], ypr[1], ypr[2]);
        const rotatedVec = matVecMul(rot, rv);
        print("Len After {} .. \n", .{abs(rotatedVec)});
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
    const sinTh = abs(k);
    const res = vc * @splat(3, cosTh) + cross(knorm, vc) * @splat(3, sinTh) + knorm * @splat(3, dot(knorm, vc) * (1 - cosTh));
    return res;
}

test "geometry. asin()" {
    print("\n asin(1)={} ", .{std.math.asin(@as(f32, 1.0))}); // 1.57079637 = π/2
    print("\n asin(-1)={}", .{std.math.asin(@as(f32, -1.0))}); // -1.57079637 = -π/2
    print("\n asin(0)={} ", .{std.math.asin(@as(f32, 0.0))}); // 0
    print("\n asin(-0)={}", .{std.math.asin(@as(f32, -0.0))}); // -0
}

pub fn testRodriguezRotation() !void {

    // check that rotating the original vector brings it to v2
    {
        var i: u32 = 0;
        while (i < 100) : (i += 1) {
            const v1 = normalize(randNormalVec3()); // start
            const v2 = normalize(randNormalVec3()); // target
            const v3 = rotateCwithRotorAB(v1, v1, v2); // try to rotate start to target
            print("Len After = {e:.3}\tDelta = {e:.3} \n", .{ abs(v3), abs(v2 - v3) });
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
            print("len(pre) {d:10.5} len(post) {d:10.5} dot(pre) {d:10.5} dot(post) {d:10.5} delta={e:13.5}\n", .{ abs(v1), abs(v3), dot1_, dot2_, dot2_ - dot1_ });
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
