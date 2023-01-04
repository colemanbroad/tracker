// HOWTO: load 'lib.a' ?
// const del = @import("/Users/broaddus/Desktop/projects-personal/zig/zig-opencl-test/src/libdelaunay.a");

const std = @import("std");
const im = @import("image_base.zig");
const geo = @import("geometry.zig");
const del = @import("delaunay.zig");

const trace = @import("trace");
pub const enable_trace = true; // must be enabled otherwise traces will be no-ops

const drawCircle = im.drawCircle;
const drawLineInBounds = im.drawLineInBounds;

const PriorityQueue = std.PriorityQueue;
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const assert = std.debug.assert;
const random = prng.random();
var prng = std.rand.DefaultPrng.init(0);

const min = std.math.min;
const max = std.math.max;

pub fn log(
    comptime message_level: std.log.Level,
    comptime scope: @Type(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    _ = scope;
    _ = message_level;
    const logfile = std.fs.cwd().createFile("timing.csv", .{ .truncate = false }) catch {
        std.debug.print(format, args);
        return;
    };
    logfile.seekFromEnd(0) catch {};
    logfile.writer().print(format, args) catch {};
    logfile.writer().writeByte(std.ascii.control_code.lf) catch {};
    logfile.close();
}

// Zig doesn't have tuple-of-types i.e. product types yet so all fields must be named. This is probably good.
// https://github.com/ziglang/zig/issues/4335

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();
// var allocator = std.testing.allocator; //(.{}){};
const Pts3 = [3]f32;
const Pts = geo.Vec2;

const test_home = "/Users/broaddus/Desktop/work/zig-tracker/test-artifacts/track/";
test {
    std.testing.refAllDecls(@This());
}

export fn add(a: i32, b: i32) i32 {
    return a + b;
}
export fn sum(a: [*]i32, n: u32) i32 {
    var tot: i32 = 0;
    var i: u32 = 0;
    while (i < n) {
        tot += a[i];
        i += 1;
        // print("Print {} me {} you fool!\n", .{tot,i});
    }
    return tot;
}

// pub fn main() !void {
test "test strain tracking" {
    print("\n\n", .{});

    const na = 101;
    const nb = 101;

    var va: [na]Pts = undefined;
    for (va) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10 };
    var vb: [nb]Pts = undefined;
    for (vb) |*v, i| v.* = .{ va[i][0] + random.float(f32) * 0.5 + 1, va[i][1] + random.float(f32) * 0.5 };

    // try mkdirIgnoreExists("strain");

    const b2a = try strainTrack(va[0..], vb[0..]);
    defer allocator.free(b2a);

    var res: [nb]i32 = undefined;

    for (b2a) |a_idx, i| {
        if (a_idx) |id| {
            res[i] = @intCast(i32, id);
        } else {
            res[i] = -1;
        }
        print("{}→{}\n", .{ i, res[i] });
    }
}

const VStatusTag = enum(u3) {
    unknown,
    parent,
    // divider, // TODO
    daughter,
    appear,
    disappear,
};

export fn strain_track2d(va: [*]f32, na: u32, vb: [*]f32, nb: u32, res: [*]i32) i32 {
    const span = trace.Span.open("strain_track2d");
    defer span.close(); // Span is closed automatically when the function returns

    const va_ = allocator.alloc(Pts, na) catch return -1;
    for (va_) |*v, i| v.* = Pts{ va[2 * i], va[2 * i + 1] };
    const vb_ = allocator.alloc(Pts, nb) catch return -1;
    for (vb_) |*v, i| v.* = Pts{ vb[2 * i], vb[2 * i + 1] };

    const b2a = strainTrack(va_, vb_) catch return -1;
    defer allocator.free(b2a);

    // Write the result to RES in-place
    for (b2a) |a_idx, i| {
        if (a_idx) |id| {
            res[i] = @intCast(i32, id);
        } else {
            res[i] = -1;
        }
    }

    return 0;
}

pub fn minmaxPts(arr: []Pts) [4]f32 {
    var min_x: f32 = 1e8;
    var min_y: f32 = 1e8;
    var max_x: f32 = -1e8;
    var max_y: f32 = -1e8;
    for (arr) |p| {
        if (p[0] < min_x) min_x = p[0];
        if (p[0] > max_x) max_x = p[0];
        if (p[1] < min_y) min_y = p[1];
        if (p[1] > max_y) max_y = p[1];
    }
    return .{ min_x, min_y, max_x, max_y };
}

fn waitForUserInput() !i64 {
    if (@import("builtin").is_test) return 0;

    const stdin = std.io.getStdIn().reader();
    const stdout = std.io.getStdOut().writer();

    var buf: [10]u8 = undefined;

    try stdout.print("Press 0 to quit: ", .{});

    if (try stdin.readUntilDelimiterOrEof(buf[0..], '\n')) |user_input| {
        const res = std.fmt.parseInt(i64, user_input, 10) catch return 1;
        if (res == 0) return 0;
    }
    return 1;
}

pub fn strainTrack(va: []Pts, vb: []Pts) ![]?u32 {

    // const stdin = std.io.getStdIn();

    const na = va.len;
    const nb = vb.len;

    var pic = try im.Img2D([4]u8).init(900, 900);
    defer pic.deinit();

    const corners = blk: {
        const cA = minmaxPts(va);
        const cB = minmaxPts(vb);
        break :blk [4]f32{ min(cA[0], cB[0]), min(cA[1], cB[1]), max(cA[2], cB[2]), max(cA[3], cB[3]) };
    };

    print("corners are: {d}\n", .{corners});

    for (va) |v| {
        const x = @floatToInt(i32, (v[0] - corners[0]) / (corners[2] - corners[0]) * (900 - 5) + 5);
        const y = @floatToInt(i32, (v[1] - corners[1]) / (corners[3] - corners[1]) * (900 - 5) + 5);
        print("x,y={},{}\n", .{ x, y });
        drawCircle([4]u8, pic, x, y, 6, .{ 200, 200, 0, 255 });
    }

    for (vb) |v| {
        const x = @floatToInt(i32, (v[0] - corners[0]) / (corners[2] - corners[0]) * (900 - 5) + 5);
        const y = @floatToInt(i32, (v[1] - corners[1]) / (corners[3] - corners[1]) * (900 - 5) + 5);
        print("x,y={},{}\n", .{ x, y });
        drawCircle([4]u8, pic, x, y, 6, .{ 50, 200, 100, 255 });
    }

    try im.saveRGBA(pic, test_home ++ "strain-test.tga");

    var a_status = try allocator.alloc(VStatusTag, na);
    defer allocator.free(a_status);
    for (a_status) |*v| v.* = .unknown;

    var b_status = try allocator.alloc(VStatusTag, nb);
    defer allocator.free(b_status);
    for (b_status) |*v| v.* = .unknown;

    const cost = try pairwiseDistances(allocator, Pts, va[0..], vb[0..]);
    defer allocator.free(cost);

    // get delaunay triangles
    // FIXME: 2D only for now
    var triangles = try del.delaunay2d(allocator, va[0..]);
    defer triangles.deinit();
    // defer allocator.free(triangles);

    // print("\n",.{});
    // for (triangles) |t,i| {
    //   print("tri {} {d}\n", .{i,t});
    //   if (i>20) break;
    // }

    // Assignment Matrix. 0 = known negative. 1 = known positive. 2 = unknown.
    // var asgn = try allocator.alloc(u8,na*nb);
    // for (asgn) |*v| v.* = 2;

    var asgn_a2b = try allocator.alloc([2]?u32, na);
    defer allocator.free(asgn_a2b);
    for (asgn_a2b) |*v| v.* = .{ null, null };

    var asgn_b2a = try allocator.alloc(?u32, nb);
    // defer allocator.free(asgn_b2a); // RETURNED BY FUNC
    for (asgn_b2a) |*v| v.* = null;

    // Count delaunay neighbours which are known, either .parent or .disappear
    var va_neib_count = try allocator.alloc(u8, na);
    defer allocator.free(va_neib_count);
    for (va_neib_count) |*v| v.* = 0;

    //
    var delaunay_array = try allocator.alloc([8]?u32, na);
    defer allocator.free(delaunay_array);
    for (delaunay_array) |*v| v.* = .{null} ** 8; // id, id, id, id, ...

    var nn_distance_ditribution = try allocator.alloc([8]?f32, na);
    defer allocator.free(nn_distance_ditribution);
    for (nn_distance_ditribution) |*v| v.* = .{null} ** 8; // id, id, id, id, ...

    // Be careful. We're iterating through triangles, so we see each interior edge TWICE!
    // This loop will de-duplicate edges.
    // for each vertex `v` from triangle `tri` with neighbour vertex `v_neib` we loop over
    // all existing neibs to see if `v_neib` already exists. if it doesn't we add it.
    // WARNING: this will break early once we hit `null`. This is fine as long as the array is front-packed like
    // [value value value null null ...]

    {
        var it = triangles.ts.iterator();
        // for (triangles) |tri| {
        while (it.next()) |kv| {
            const tri = kv.key_ptr.*;
            for (tri) |v, i| {
                outer: for ([3]u32{ 0, 1, 2 }) |j| {
                    if (i == j) continue;
                    const v_neib = tri[j];
                    for (delaunay_array[v]) |v_neib_existing, k| {
                        if (v_neib_existing == null) {
                            delaunay_array[v][k] = v_neib;
                            nn_distance_ditribution[v][k] = dist(Pts, va[v], va[v_neib]); // squared euclidean
                            continue :outer;
                        }
                        if (v_neib_existing.? == v_neib) continue :outer;
                    }
                }
            }
        }
    }

    const avgdist = blk: {
        var ad: f32 = 0;
        var count: u32 = 0;
        for (nn_distance_ditribution) |nnd| {
            for (nnd) |dq| {
                if (dq) |d| {
                    ad += d;
                    count += 1;
                }
            }
        }
        break :blk ad / @intToFloat(f32, count);
    };

    // print("\n",.{});
    // for (delaunay_array) |da| {
    //   print("{d}\n", .{da});
    // }

    // for each pair of vertices v0,v1 on a delaunay edge we have an associated cost for their translation difference
    // we also have a cost based on the displacement v0(t),v0(t+1)
    // we can pick a vertex at random and choose it's lowest cost match. then given that assignment we can fill in the rest.

    var vertQ = PriorityQueue(TNeibsAssigned, void, gtTNeibsAssigned).init(allocator, {});
    defer vertQ.deinit();
    try vertQ.add(.{ .idx = 50, .nneibs = 0 });

    // Greedily make assignments for each vertex in the queue based based on minimum strain cost
    // Select vertices by largest number of already-assigned-neighbours
    while (vertQ.count() > 0) {

        // this is the next vertex to match
        const v = vertQ.remove();

        // FIXME allow for divisions
        switch (a_status[v.idx]) {
            .parent => continue,
            .disappear => continue,
            else => {},
        }

        // find best match from among all vb based on strain costs
        var bestcost: ?f32 = null;
        var bestidx: ?usize = null;

        // TODO: replace linear lookup with O(1) GridHash
        for (vb) |x_vb, vb_idx| {

            // skip vb if already assigned
            if (b_status[vb_idx] == .daughter) continue;

            const x_va = va[v.idx];
            const nn_cost: f32 = dist(Pts, x_va, x_vb);

            // avoid long jumps
            if (nn_cost > avgdist * 2) continue;

            // compute strain cost
            const dx = x_vb - x_va;
            var dx_cost: f32 = 0;
            for (delaunay_array[v.idx]) |va_neib_idx| {
                const a_idx = if (va_neib_idx) |_v| _v else continue;
                if (a_status[a_idx] != .parent) continue;

                const b_idx = asgn_a2b[a_idx][0].?;
                const dx_va_neib = vb[b_idx] - va[a_idx];
                dx_cost += dist(Pts, dx, dx_va_neib);
            }

            // cost=0 for first vertex in queue (no neibs). then use nearest-neib cost.
            if (dx_cost == 0) {
                // add velgrad cost (if any exist)
                if (v.idx == 0) {
                    print("va_idx={} , vb_idx={}\n", .{ v.idx, vb_idx });
                    print("bingo dog: {}\n", .{nn_cost});
                }
                dx_cost = nn_cost;
            }

            if (bestcost == null or dx_cost < bestcost.?) {
                bestcost = dx_cost;
                bestidx = vb_idx;
            }
        }

        if (v.idx == 0) {
            // print("va_idx={} , vb_idx={}\n",.{v.idx,vb_idx});
            // print("bingo dog: {}\n",.{nn_cost});
            print("best: {any} {any}\n", .{ bestcost, bestidx });
        }

        if (v.idx == 50) {
            bestidx = 50;
        }

        // update cell status and graph relations
        if (bestidx) |b_idx| {
            a_status[v.idx] = .parent;
            b_status[b_idx] = .daughter;
            asgn_a2b[v.idx][0] = @intCast(u32, b_idx);
            asgn_b2a[b_idx] = v.idx;

            // draw a line
            const xa = @floatToInt(i32, (va[v.idx][0] - corners[0]) / (corners[2] - corners[0]) * (900 - 5) + 5);
            const ya = @floatToInt(i32, (va[v.idx][1] - corners[1]) / (corners[3] - corners[1]) * (900 - 5) + 5);
            const xb = @floatToInt(i32, (vb[b_idx][0] - corners[0]) / (corners[2] - corners[0]) * (900 - 5) + 5);
            const yb = @floatToInt(i32, (vb[b_idx][1] - corners[1]) / (corners[3] - corners[1]) * (900 - 5) + 5);
            drawLineInBounds([4]u8, pic, xa, ya, xb, yb, .{ 255, 255, 255, 255 });
        } else {
            a_status[v.idx] = .disappear;
        }

        // add delaunay neibs of `v` to the PriorityQueue
        for (delaunay_array[v.idx]) |va_neib| {
            if (va_neib == null) continue;
            va_neib_count[va_neib.?] += 1;
            try vertQ.add(.{ .idx = va_neib.?, .nneibs = va_neib_count[va_neib.?] });
        }

        try im.saveRGBA(pic, test_home ++ "strain-test-2.tga");
        const userval = try waitForUserInput();
        if (userval == 0) break;
    }

    var hist: [5]u32 = .{0} ** 5;
    for (a_status) |s| {
        hist[@enumToInt(s)] += 1;
    }

    print("\n", .{});
    print("Assignment Histogram\n", .{});
    for (hist) |h, i| {
        const e = @intToEnum(VStatusTag, i);
        print("{s}→{d}\n", .{ @tagName(e), h });
    }

    // print("The Assignments are...\n", .{});
    // for (asgn_a2b) |b_idx,a_idx| {
    //   print("{d}→{d} , {} \n", .{a_idx,b_idx, a_status[a_idx]});
    // }

    return asgn_b2a;
}

test "test track. greedy Strain Tracking 2D" {
    print("\n\n", .{});

    const na = 101;
    const nb = 102;

    var va: [na]Pts = undefined;
    for (va) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10 };
    var vb: [nb]Pts = undefined;
    for (vb) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10 };

    const b2a = try strainTrack(va[0..], vb[0..]);
    defer allocator.free(b2a);
}

fn argmax1d(comptime T: type, arr: []T) struct { max: T, idx: usize } {
    var amax = arr[0];
    var idx: usize = 0;
    for (arr) |v, i| {
        if (v > amax) {
            idx = i;
            amax = v;
        }
    }
    return .{ .max = amax, .idx = idx };
}

test "test track. greedy min-cost tracking 3D" {
    print("\n\n", .{});

    const na = 1001;
    const nb = 1002;

    var va: [na]Pts3 = undefined;
    for (va) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10, random.float(f32) * 10 };
    var vb: [nb]Pts3 = undefined;
    for (vb) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10, random.float(f32) * 10 };

    const parents = try greedyTrack(Pts3, va[0..], vb[0..]);
    defer allocator.free(parents);
}

test "test track. greedy min-cost tracking 2D" {
    print("\n\n", .{});

    const na = 101;
    const nb = 102;

    var va: [na]Pts = undefined;
    for (va) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10 };
    var vb: [nb]Pts = undefined;
    for (vb) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10 };

    const parents = try greedyTrack(Pts, va[0..], vb[0..]);
    defer allocator.free(parents);
}

export fn greedy_track2d(va: [*]f32, na: u32, vb: [*]f32, nb: u32, res: [*]i32) i32 {
    const va_ = allocator.alloc(Pts, na) catch return -1;
    for (va_) |*v, i| v.* = Pts{ va[2 * i], va[2 * i + 1] };
    const vb_ = allocator.alloc(Pts, nb) catch return -1;
    for (vb_) |*v, i| v.* = Pts{ vb[2 * i], vb[2 * i + 1] };

    const parents = greedyTrack(Pts, va_, vb_) catch return -1;
    defer allocator.free(parents);

    for (parents) |p, i| res[i] = p;

    return 0;
}

pub fn greedyTrack(comptime T: type, va: []T, vb: []T) ![]i32 {
    const na = va.len;
    const nb = vb.len;

    // count number of out edges on A
    var aout = try allocator.alloc(u8, na);
    defer allocator.free(aout);
    for (aout) |*v| v.* = 0;

    // count number of in edges on B
    var bin = try allocator.alloc(u8, nb);
    defer allocator.free(bin);
    for (bin) |*v| v.* = 0;

    const cost = try pairwiseDistances(allocator, T, va[0..], vb[0..]);
    defer allocator.free(cost);

    var asgn = try allocator.alloc(u8, na * nb);
    defer allocator.free(asgn);
    for (asgn) |*v| v.* = 0;

    // sort costs
    // continue adding costs cheapest-first as long as they don't violate asgn constraints
    // sort each row by smallest cost? then
    var edgeQ = PriorityQueue(CostEdgePair, void, ltCostEdgePair).init(allocator, {});
    defer edgeQ.deinit();
    for (cost) |c, i| {
        const ia = i / nb;
        const ib = i % nb;
        try edgeQ.add(.{ .cost = c, .ia = @intCast(u32, ia), .ib = @intCast(u32, ib) });
    }

    // greedily go through edges and add them to graph iff they don't violate constraints
    var count: usize = 0;
    while (true) {
        count += 1;
        if (count == na * nb + 1) break;

        const edge = edgeQ.remove();
        // print("Cost {}, Edge {} {}\n", edge);

        if (aout[edge.ia] == 2 or bin[edge.ib] == 1) continue;
        asgn[edge.ia * nb + edge.ib] = 1;

        aout[edge.ia] += 1;
        bin[edge.ib] += 1;
    }

    var parents = try allocator.alloc(i32, nb);
    for (vb) |_, i| {
        for (va) |_, j| {
            // if the edge is active, then assign and break (can only have 1 parent max)
            if (asgn[j * nb + i] == 1) {
                parents[i] = @intCast(i32, j);
                break;
            }
            // otherwise there is no parent
            parents[i] = -1;
        }
    }

    for (parents) |p, i| {
        print("{d} → {d}\n", .{ i, p });
        if (i > 10) break;
    }

    return parents;
}

const CostEdgePair = struct { cost: f32, ia: u32, ib: u32 };
fn ltCostEdgePair(context: void, a: CostEdgePair, b: CostEdgePair) std.math.Order {
    _ = context;
    return std.math.order(a.cost, b.cost);
}

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
        Pts => (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]),
        Pts3 => (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) + (x[2] - y[2]) * (x[2] - y[2]),
        else => unreachable,
    };
}

const TNeibsAssigned = struct { idx: u32, nneibs: u8 };
fn gtTNeibsAssigned(context: void, a: TNeibsAssigned, b: TNeibsAssigned) std.math.Order {
    _ = context;
    return std.math.order(a.nneibs, b.nneibs);
}
