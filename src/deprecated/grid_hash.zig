
const std = @import("std");
const im = @import("image_base.zig");
const geo = @import("geometry.zig");

const print = std.debug.print;
const assert = std.debug.assert;
const expect = std.testing.expect;
// var allocator = std.testing.allocator;
var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

const Allocator = std.mem.Allocator;
const Vec2 = geo.Vec2;
const Vec3 = geo.Vec3;
const Range = geo.Range;
const BBox = geo.BBox;

const clipi = geo.clipi;

const test_home = "../test-artifacts/grid_hash/";

test {
    std.testing.refAllDecls(@This());
}

/// run all experiments
pub fn main() !void {
    try test1();
    try test2();
    try test3();
    try test4();
}



fn randomFloats(comptime n:u32) [n]f32 {
    var x:[n]f32 = undefined;
    for (x) |*v| v.* = random.float(f32) * 100.0;
    return x;
}


/// test spatial. bin random points onto 2D grid
fn test1() !void {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();
    var allo = allocator.allocator();
    im.allocator = allo; // Now image_base functions also use the Arena?! [this is bad practice as it hides allocations 😬]

    const xs = randomFloats(100); // [0,100)
    const ys = randomFloats(100);
    var grid = try im.Img2D(u8).init(10, 10);

    for (xs) |x, i| {
        const y = ys[i];
        const nx = @floatToInt(usize, x / 10);
        const ny = @floatToInt(usize, y / 10);
        grid.img[nx * 10 + ny] += 1;
    }

    var rgba = try im.Img2D([4]u8).init(10, 10);

    for (grid.img) |g, i| {
        const r = @intCast(u8, (@intCast(u16, g) * 10) % 255);
        rgba.img[i] = .{ r, r, r, 255 };
    }
    try im.saveRGBA(rgba, test_home ++ "test1.tga");
}

/// test spatial. GridHash
fn test2() !void {
    var pts: [100]Vec2 = undefined;
    for (pts) |*v| v.* = .{ random.float(f32) * 100.0, random.float(f32) * 100.0 };

    var gh = try GridHash.init(allocator, 10, 10, 6, pts[0..], null);
    defer gh.deinit();
    for (pts) |p| {
        _ = gh.neibs(p);
        print("{}→{d}\n",.{i,gh.neibs(p)});
    }
}

// Prealloc idx and dist memory for fast multiple queries
const IdsDists = struct {
    ids: []u16,
    dists: []f32,
    n_ids: *usize,

    pub fn init(al: Allocator, capacity: u16) !IdsDists {
        var ids = try al.alloc(u16, capacity);
        var dists = try al.alloc(f32, capacity);
        var n_ids: usize = capacity;
        return IdsDists{ .ids = ids, .dists = dists, .n_ids = &n_ids };
    }

    pub fn deinit(self: IdsDists, al: Allocator) void {
        al.free(self.ids);
        al.free(self.dists);
    }
};

const GridHash = struct {
    const Self = @This();
    const Elem = ?u16;
    const SetRoot = u16;

    map: []Elem,
    nx: u32,
    ny: u32,
    nelemax: u32,
    allo: Allocator,
    bb: geo.BBox,
    pts: []Vec2,

    pub fn init(
        allo: Allocator,
        nx: u32,
        ny: u32,
        nelemax: u32,
        _pts: []Vec2,
        labels: ?[]u16,
    ) !Self {
        var pts = try allo.alloc(Vec2, _pts.len);
        for (pts) |*p, i| p.* = _pts[i];
        const map = try allocator.alloc(Elem, nx * ny * nelemax);
        // print("\n\nmap.len = {}\n",.{map.len});
        errdefer allocator.free(map);
        for (map) |*v| v.* = null;

        var bb = geo.boundsBBox(pts);
        const ddx = 0.05 * (bb.x.hi - bb.x.lo);
        const ddy = 0.05 * (bb.y.hi - bb.y.lo);

        bb.x.lo += -ddx;
        bb.x.hi += ddx;
        bb.y.lo += -ddy;
        bb.y.hi += ddy;

        // const dx = (bb.x.hi-bb.x.lo)/@intToFloat(f32,nx);
        // const dy = (bb.y.hi-bb.y.lo)/@intToFloat(f32,ny);

        outer: for (pts) |p, i| { // i = pt label

            const l = if (labels) |lab| lab[i] else i;

            const ix = x2grid(p[0], bb.x, nx);
            const iy = x2grid(p[1], bb.y, ny);

            const idx = (ix * ny + iy) * nelemax;
            // print("ix,iy,i,idx = {},{},{},{}\n", .{ix,iy,i,idx});

            for (map[idx .. idx + nelemax]) |*m| {
                if (m.* == null) {
                    m.* = @intCast(u16, l);
                    continue :outer;
                }
            }

            return error.PointDensityError;
        }

        return Self{
            .allo = allo,
            .map = map,
            .nx = nx,
            .ny = ny,
            .nelemax = nelemax,
            // .dx=dx,
            // .dy=dy,
            .bb = bb,
            .pts = pts,
        };
    }

    pub fn deinit(self: Self) void {
        self.allo.free(self.map);
        self.allo.free(self.pts);
    }

    fn x2grid(x: f32, xr: Range, nx: u32) u32 {
        const dx = (xr.hi - xr.lo) / @intToFloat(f32, nx);
        // print("floor((x-xr.lo)/dx) = {}\n", .{floor((x-xr.lo)/dx)});
        const ix = @floatToInt(i32, @floor((x - xr.lo) / dx));
        if (ix < 0) return 0;
        if (ix > nx - 1) return nx - 1;
        return @intCast(u32, ix);
    }

    /// returns slice of map. do not modify!
    pub fn neibs(self: Self, p: Vec2) []Elem {
        // const ixiy = pt2grid(p,self.bb,self.nx,self.ny);
        // const ix=ixiy[0];
        // const iy=ixiy[1];
        const ix = x2grid(p[0], self.bb.x, self.nx);
        const iy = x2grid(p[1], self.bb.y, self.ny);
        // print("neibs ix iy {} {}\n", .{ix,iy});
        const idx = (ix * self.ny + iy) * self.nelemax;
        return self.map[idx .. idx + self.nelemax];
    }

    // first search pairwise in grid, then if need more points expand to surroundings.
    // search all boxes within `radius` of `p`
    pub fn nnRadius(self: Self, res: IdsDists, p: Vec2, radius: f32) !void {
        const ix_min = x2grid(p[0] - radius, self.bb.x, self.nx);
        const ix_max = x2grid(p[0] + radius, self.bb.x, self.nx);
        const iy_min = x2grid(p[1] - radius, self.bb.y, self.ny);
        const iy_max = x2grid(p[1] + radius, self.bb.y, self.ny);

        // const nx = ix_max-ix_min+1;
        // const ny = iy_max-iy_min+1;

        var nn_ids = res.ids;
        var dists = res.dists;

        var nn_count: usize = 0;
        var xid = ix_min;
        while (xid <= ix_max) : (xid += 1) {
            var yid = iy_min;
            while (yid <= iy_max) : (yid += 1) {
                const idx = (xid * self.ny + yid) * self.nelemax;
                // print("idx {} \n", .{idx});
                const bin = self.map[idx .. idx + self.nelemax];
                // print("p {} xid {} yid {} bin {d}\n", .{p,xid,yid,bin});
                // print("neibs {d} \n", .{self.neibs(p)});
                for (bin) |e_| {
                    const e = if (e_) |e| e else continue;
                    const pt_e = self.pts[e];
                    // const delta = p-pt_e;
                    // const mydist = @sqrt(@reduce(.Add,delta*delta));
                    const mydist = dist(Vec2, p, pt_e);

                    if (mydist < radius) {
                        // print("adding e={}\n",.{e});
                        nn_ids[nn_count] = e;
                        dists[nn_count] = mydist;
                        nn_count += 1;
                    }
                }
            }
        }

        res.n_ids.* = nn_count;

        // remove undefined regions
        // nn_ids = al.shrink(nn_ids,nn_count);
        // dists = al.shrink(dists,nn_count);
        // return IdsDists{.ids=nn_ids , .dists=dists};
    }

    // Write in-place to res_elems. no alloc required. assert k=knn.len
    // pub fn knn(self: Self, p: Vec2, comptime k: u8, res_elems: []u16) !void {
    //     const ix_start = x2grid(p[0], self.bb.x, self.nx);
    //     const iy_start = x2grid(p[1], self.bb.y, self.ny);

    //     var elems: [2 * k]u16 = undefined;
    //     var dists: [2 * k]f32 = undefined;

    //     var nn_count: usize = 0;
    //     var k_count: u8 = 0;
    //     var xid = ix_start;

    //     var boxQ = std.Queue([2]u16); // box id

    //     while (!boxQ.isEmpty()) {
    //         var yid = iy_start;
    //         const idx = (xid * self.ny + yid) * self.nelemax;
    //         // print("idx {} \n", .{idx});
    //         const bin = self.map[idx .. idx + self.nelemax];
    //         // print("p {} xid {} yid {} bin {d}\n", .{p,xid,yid,bin});
    //         // print("neibs {d} \n", .{self.neibs(p)});
    //         for (bin) |e_| {
    //             const e = if (e_) |e| e else continue;
    //             const pt_e = self.pts[e];
    //             // const delta = p-pt_e;
    //             // const mydist = @sqrt(@reduce(.Add,delta*delta));
    //             const mydist = dist(Vec2, p, pt_e);

    //             if (mydist < radius) {
    //                 // print("adding e={}\n",.{e});
    //                 nn_ids[nn_count] = e;
    //                 dists[nn_count] = mydist;
    //                 nn_count += 1;
    //             }
    //         }
    //     }
    // }

    // res.n_ids.* = nn_count;
    //   }
};

// array order is [a,b]. i.e. a has stride nb. b has stride 1.
pub fn pairwiseDistances(al: Allocator, comptime T: type, a: []T, b: []T) ![]f32 {
    const na = a.len;
    const nb = b.len;

    var cost = try al.alloc(f32, na * nb);
    for (cost) |*v| v.* = 0;

    for (a) |x, i| {
        for (b) |y, j| {
            cost[i * nb + j] = dist(T, x, y);
        }
    }

    return cost;
}

fn dist(comptime T: type, x: T, y: T) f32 {
    return switch (T) {
        Vec2 => (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]),
        Vec3 => (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) + (x[2] - y[2]) * (x[2] - y[2]),
        else => unreachable,
    };
}

// test "test spatial. radius neibs" {
fn test3() !void {
    const N = 5_000;
    var pts: [N]Vec2 = undefined;
    for (pts) |*v| v.* = .{ random.float(f32) * 100.0, random.float(f32) * 100.0 };

    const sqrtN = @floatToInt(u16, @sqrt(@intToFloat(f32, N)));
    var gh = try GridHash.init(allocator, sqrtN, sqrtN, 20, pts[0..], null);
    defer gh.deinit();

    // Prealloc idx and dist memory for fast multiple queries
    const res = try IdsDists.init(allocator, 200);
    defer res.deinit(allocator);

    // Prealloc pdneibs buffer;
    const pairdist = try pairwiseDistances(allocator, Vec2, pts[0..], pts[0..]);
    defer allocator.free(pairdist);

    var buf = try allocator.alloc(u16, 200);
    defer allocator.free(buf);

    for (pts) |p, i| {
        try gh.nnRadius(res, p, 1.0);
        var res_view = res.ids[0..res.n_ids.*]; // leave res.ids length const. Change size of view.

        const pdneibs = blk: {
            var count: u16 = 0;
            for (pts) |_, j| {
                if (pairdist[i * N + j] < 1.0) {
                    buf[count] = @intCast(u16, j);
                    count += 1;
                }
            }
            break :blk buf[0..count];
        };

        // Now compare the results
        std.sort.sort(u16, res_view, {}, comptime std.sort.asc(u16));
        // print("{d}...{d}\n", .{pdneibs,s.ids});
        std.testing.expect(std.mem.eql(u16, pdneibs, res_view)) catch {
            print("\n\nERROR\n\npd={d} , res.ids={d}\n\n", .{ pdneibs, res_view });
            unreachable;
        };
    }
}

// test "test spatial. radius speed test" {
fn test4() !void {
    var alltimes: [4][6]i128 = undefined;

    for (alltimes) |*timez| {
        const N = 5_000;
        var pts: [N]Vec2 = undefined;
        for (pts) |*v| v.* = .{ random.float(f32) * 100.0, random.float(f32) * 100.0 };

        var qpts: [N]Vec2 = undefined;
        for (qpts) |*v| v.* = .{ random.float(f32) * 100.0, random.float(f32) * 100.0 };

        const t0 = std.time.nanoTimestamp();

        const sqrtN = @floatToInt(u16, @sqrt(@intToFloat(f32, N)));
        var gh = try GridHash.init(allocator, sqrtN, sqrtN, 20, pts[0..], null);
        defer gh.deinit();

        const t1 = std.time.nanoTimestamp();

        // Prealloc idx and dist memory for fast multiple queries
        var res = try IdsDists.init(allocator, 200);
        defer res.deinit(allocator);

        const t1a = std.time.nanoTimestamp();

        for (qpts) |q| {
            _ = try gh.nnRadius(res, q, 1.0);
        }

        const t2 = std.time.nanoTimestamp();

        // const pairdist = try pairwise_distances(allocator,Vec2,pts[0..],pts[0..]);
        // defer allocator.free(pairdist);

        const t3 = std.time.nanoTimestamp();

        var buf = try allocator.alloc(u16, 200);
        defer allocator.free(buf);

        for (qpts) |q| {
            _ = blk: {
                var count: u16 = 0;
                for (pts) |p, j| {
                    // if (pairdist[i*N + j] < 1.0) {
                    const dx = q - p;
                    const pq_dist = @sqrt(@reduce(.Add, dx * dx));
                    if (pq_dist < 1.0) {
                        // print("pqd = {}   ", .{pq_dist});
                        buf[count] = @intCast(u16, j);
                        count += 1;
                    }
                }
                // buf = allocator.shrink(buf,count);
                break :blk buf;
            };
            // defer allocator.free(neibs);
        }

        const t4 = std.time.nanoTimestamp();
        const times = .{ t0 - t0, t1 - t0, t1a - t0, t2 - t0, t3 - t0, t4 - t0 };
        timez.* = times;
    }

    print("\nTiming\n", .{});
    for (alltimes) |_, i| {
        print("\n", .{});
        for (alltimes[0]) |_, j| {
            const t = @intToFloat(f32, alltimes[i][j]) / 1e6;
            print("{d:.3} ", .{t});
        }
    }
    print("\n\n", .{});

    // var mean:[5]f32 = undefined;
    // for (alltimes) |_,i| {
    // for (alltimes[0]) |_,j| {
    //     mean[i] += alltimes[i][j];
    // }
    // }
    // for (mean) |*v,i| v.* /= alltimes[0].len;

    // var stddev:[5]f32 = undefined;
    // for (alltimes) |_,i| {
    // for (alltimes[0]) |_,j| {
    //     const x = alltimes[i][j] - mean[i];
    //     std[i] += x*x;
    // }
    // }

    // stddev[i]
    // for (stddev) |*s,i| s.* -= mean[i]*mean[i];

    // print("\n\ntimings...{d}\n", .{alltimes});
    // print("\n\nmean...{d}\n", .{mean});
    // print("\n\nstd...{d}\n", .{stddev});
}
