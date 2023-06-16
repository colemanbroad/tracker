// HOWTO: load 'lib.a' ?
// const del = @import("/Users/broaddus/Desktop/projects-personal/zig/zig-opencl-test/src/libdelaunay.a");

const std = @import("std");
const im = @import("image_base.zig");
const geo = @import("geometry.zig");
const del = @import("delaunay.zig");

// const trace = @import("trace");
// pub const enable_trace = true; // must be enabled otherwise traces will be no-ops

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
    const logfile = std.fs.cwd().createFile("trace.tracker.csv", .{ .truncate = false }) catch {
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
var allocator = gpa.allocator();
// var allocator = std.testing.allocator; //(.{}){};

const Pts3 = [3]f32;
const Pts = geo.Vec2;

const test_home = "/Users/broaddus/Desktop/work/isbi/zig-tracker/test-artifacts/track/";

test {
    // std.testing.refAllDecls(@This());
}

// pub fn main() !void {
test "test strain tracking" {
    print("\n\n", .{});

    const na = 101;
    const nb = 101;

    var va: [na]Pts = undefined;
    for (&va) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10 };
    var vb: [nb]Pts = undefined;
    for (&vb, 0..) |*v, i| v.* = .{ va[i][0] + random.float(f32) * 0.5 + 1, va[i][1] + random.float(f32) * 0.5 };

    // try mkdirIgnoreExists("strain");

    const b2a = try strainTrack(va[0..], vb[0..]);
    defer allocator.free(b2a);

    var res: [nb]i32 = undefined;

    for (b2a, 0..) |a_idx, i| {
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
    // const span = trace.Span.open("strain_track2d");
    // defer span.close(); // Span is closed automatically when the function returns

    const va_ = allocator.alloc(Pts, na) catch return -1;
    for (va_, 0..) |*v, i| v.* = Pts{ va[2 * i], va[2 * i + 1] };
    const vb_ = allocator.alloc(Pts, nb) catch return -1;
    for (vb_, 0..) |*v, i| v.* = Pts{ vb[2 * i], vb[2 * i + 1] };

    const b2a = strainTrack(va_, vb_) catch return -1;
    defer allocator.free(b2a);

    // Write the result to RES in-place
    for (b2a, 0..) |a_idx, i| {
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

// Track from va -> vb using velocity gradient tracking.
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
            for (tri, 0..) |v, i| {
                outer: for ([3]u32{ 0, 1, 2 }) |j| {
                    if (i == j) continue;
                    const v_neib = tri[j];
                    for (delaunay_array[v], 0..) |v_neib_existing, k| {
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
        for (vb, 0..) |x_vb, vb_idx| {

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
    for (hist, 0..) |h, i| {
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
    for (&va) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10 };
    var vb: [nb]Pts = undefined;
    for (&vb) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10 };

    const b2a = try strainTrack(va[0..], vb[0..]);
    defer allocator.free(b2a);
}

fn argmax1d(comptime T: type, arr: []T) struct { max: T, idx: usize } {
    var amax = arr[0];
    var idx: usize = 0;
    for (arr, 0..) |v, i| {
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
    for (&va) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10, random.float(f32) * 10 };
    var vb: [nb]Pts3 = undefined;
    for (&vb) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10, random.float(f32) * 10 };

    const parents = try greedyTrack(Pts3, va[0..], vb[0..]);
    defer allocator.free(parents);
}

test "test track. greedy min-cost tracking 2D" {
    print("\n\n", .{});

    // allocator = std.testing.allocator;

    const na = 101;
    const nb = 102;

    var va: [na]Pts = undefined;
    for (&va) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10 };
    var vb: [nb]Pts = undefined;
    for (&vb) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10 };

    // @breakpoint();

    const parents = try greedyTrack(Pts, va[0..], vb[0..]);
    defer allocator.free(parents);
}

export fn greedy_track2d(va: [*]f32, na: u32, vb: [*]f32, nb: u32, res: [*]i32) i32 {
    const va_ = allocator.alloc(Pts, na) catch return -1;
    for (va_, 0..) |*v, i| v.* = Pts{ va[2 * i], va[2 * i + 1] };
    const vb_ = allocator.alloc(Pts, nb) catch return -1;
    for (vb_, 0..) |*v, i| v.* = Pts{ vb[2 * i], vb[2 * i + 1] };

    const parents = greedyTrack(Pts, va_, vb_) catch return -1;
    defer allocator.free(parents);

    for (parents, 0..) |p, i| res[i] = p;

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
    for (cost, 0..) |c, i| {
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
    for (vb, 0..) |_, i| {
        for (va, 0..) |_, j| {
            // if the edge is active, then assign and break (can only have 1 parent max)
            if (asgn[j * nb + i] == 1) {
                parents[i] = @intCast(i32, j);
                break;
            }
            // otherwise there is no parent
            parents[i] = -1;
        }
    }

    for (parents, 0..) |p, i| {
        print("{d} → {d}\n", .{ i, p });
        if (i > 10) break;
    }

    return parents;
}

const Parent = enum { missing, exists };
const ParentIdx = union(Parent) {
    missing: void,
    exists: usize,
};
const Assignment = struct {
    pc1: []Pts,
    pc2: []Pts,
    parents: []ParentIdx, // parent.len == pc2.len and parent[idx] in pc1 with idx in pc2.
};

pub fn nearestParentAssignment_NNSorted(pc1: []Pts, pc2: []Pts) ![]ParentIdx {
    var assignment = try allocator.alloc(ParentIdx, pc2.len);

    const index = blk: {
        var from_sorted = try allocator.alloc(usize, pc1.len);
        for (from_sorted, 0..) |*v, i| {
            v.* = i;
        }

        const lt = struct {
            fn lt(pts: []Pts, idx0: usize, idx1: usize) bool {
                if (pts[idx0][0] < pts[idx1][0]) return true;
                return false;
            }
        }.lt;

        std.sort.heap(usize, from_sorted, pc1, lt);

        // Create the inverse permutation
        var to_sorted = try allocator.alloc(usize, pc1.len);
        for (from_sorted, 0..) |idx, i| {
            to_sorted[idx] = i;
        }

        break :blk .{ .from_sorted = from_sorted, .to_sorted = to_sorted };
    };

    const pc1_sorted = blk: {
        var pc1_sorted = try allocator.alloc([2]f32, pc1.len);
        for (pc1_sorted, index.from_sorted) |*v, idx| {
            v.* = pc1[idx];
        }
        break :blk pc1_sorted;
    };

    for (pc2, 0..) |p_child, i| {
        const nearest_parent_idx_sorted = nn_tools.findNearestNeibFromSortedList(pc1_sorted, p_child);
        assignment[i] = ParentIdx{ .exists = index.from_sorted[nearest_parent_idx_sorted] };
    }
    return assignment;
}

pub fn nearestParentAssignment_NNBrute(pc1: []Pts, pc2: []Pts) ![]ParentIdx {
    var assignment = try allocator.alloc(ParentIdx, pc2.len);

    var pc1_f32 = try allocator.alloc([2]f32, pc1.len);
    for (pc1_f32, pc1) |*v, x| v.* = x;

    for (pc2, 0..) |p_child, i| {
        const nearest_parent_idx_sorted = nn_tools.findNearestNeibBruteForce(pc1_f32, p_child);
        assignment[i] = ParentIdx{ .exists = nearest_parent_idx_sorted };
    }
    return assignment;
}

// Iterate over consecutive timepoints in Tracking and connect points to
// nearest parent in the previous frame.
pub fn nearestParentAssignment(tracking: Tracking2D) !void {

    // sort by (time, x-coord)
    const lt = struct {
        fn lt(ctx: void, t0: PtIdTimeParent, t1: PtIdTimeParent) bool {
            _ = ctx;
            if (t0.time < t1.time) return true;
            if (t0.time == t1.time and t0.pt[0] < t1.pt[0]) return true;
            return false;
        }
    }.lt;

    std.sort.heap(PtIdTimeParent, tracking, {}, lt);

    // then find time boundaries
    const TimeCount = struct { time: u16, count: u16, cum: u32 };
    // Can't be longer than this. times are discrete and >= 0.
    var timebounds = try allocator.alloc(TimeCount, tracking[tracking.len - 1].time + 1);
    defer allocator.free(timebounds);

    for (timebounds) |*v| {
        v.count = 0;
        v.cum = 0;
    }

    // initialize with the first object
    timebounds[0].time = tracking[0].time;
    timebounds[0].count += 1;
    timebounds[0].cum += 1;

    // fill up time bounds
    {
        var tb_idx: u16 = 0;
        for (1..tracking.len) |tr_idx| {
            if (tracking[tr_idx].time > tracking[tr_idx - 1].time) {
                tb_idx += 1;
                timebounds[tb_idx].time = tracking[tr_idx].time;
                timebounds[tb_idx].cum = timebounds[tb_idx - 1].cum;
            }
            timebounds[tb_idx].count += 1;
            timebounds[tb_idx].cum += 1;
        }
    }

    // then connect children to nearest parent in prev frame
    var t0_start: u32 = 0;
    for (0..timebounds.len - 1) |tb_idx| {
        const t0_end = timebounds[tb_idx].cum;
        const t1_end = timebounds[tb_idx + 1].cum;
        defer t0_start = t0_end;

        const trackslice_prev = tracking[t0_start..t0_end];
        const trackslice_curr = tracking[t0_end..t1_end];

        // Clear the screen. Draw prev pts in green, curr pts in blue.
        // Make connections one at a time in red. Draw the GT connection in teal.

        for (win.pix.img) |*v| v.* = .{ 0, 0, 0, 255 };
        for (trackslice_prev) |p| {
            const x = @floatToInt(i32, p.pt[0] * 750 + 25);
            const y = @floatToInt(i32, p.pt[1] * 750 + 25);
            // const y = @floatToInt(i32, @intToFloat(f32, p.time) / 30 * 750 + 25);
            im.drawCircle([4]u8, win.pix, x, y, 3, .{ 0, 255, 0, 255 });
        }
        for (trackslice_curr) |p| {
            const x = @floatToInt(i32, p.pt[0] * 750 + 25);
            const y = @floatToInt(i32, p.pt[1] * 750 + 25);
            // const y = @floatToInt(i32, @intToFloat(f32, p.time) / 30 * 750 + 25);
            im.drawCircle([4]u8, win.pix, x, y, 3, .{ 255, 0, 0, 255 });
        }

        for (trackslice_curr) |*p| {
            {
                const x = @floatToInt(i32, p.pt[0] * 750 + 25);
                const y = @floatToInt(i32, p.pt[1] * 750 + 25);
                if (p.parent_id) |pid| {
                    const parent = blk: {
                        for (tracking) |t| {
                            if (t.id == pid) break :blk t;
                        }
                        unreachable;
                    };
                    const x2 = @floatToInt(i32, parent.pt[0] * 750 + 25);
                    const y2 = @floatToInt(i32, parent.pt[1] * 750 + 25);
                    im.drawLineInBounds([4]u8, win.pix, x, y, x2, y2, .{ 255, 255, 0, 255 });
                }
            }

            const parent_idx = nn_tools.findNearestNeibFromSortedListGeneric(PtIdTimeParent, trackslice_prev, p.pt);
            const parent = trackslice_prev[parent_idx];
            p.parent_id = parent.id;

            const x = @floatToInt(i32, p.pt[0] * 750 + 25);
            const y = @floatToInt(i32, p.pt[1] * 750 + 25);
            const x2 = @floatToInt(i32, parent.pt[0] * 750 + 25);
            const y2 = @floatToInt(i32, parent.pt[1] * 750 + 25);
            // const y2 = @floatToInt(i32, @intToFloat(f32, parent_node.time) / 30 * 750 + 25);
            // im.drawCircle([4]u8, win.pix, x2, y2, 3, .{ 0, 0, 255, 255 });
            im.drawLineInBounds([4]u8, win.pix, x, y, x2, y2, .{ 0, 0, 255, 255 });
            win.awaitKeyPressAndUpdateWindow();
        }
    }
}

pub fn main() !void {
    const na = 101;
    const nb = 101;
    var va: [na]Pts = undefined;
    for (&va) |*v| v.* = .{ random.float(f32) * 10, random.float(f32) * 10 };
    var vb: [nb]Pts = undefined;
    for (&vb, 0..) |*v, i| v.* = .{
        va[i][0] + random.float(f32) * 0.05,
        va[i][1] + random.float(f32) * 0.05,
    };

    const greedy_assignment = try nearestParentAssignment_NNBrute(&va, &vb);
    // const greedy_assignment = try nearestParentAssignment(&va, &vb);

    print("List of parents \n", .{});
    for (0.., greedy_assignment) |idx, par| {
        print("{d}→{d} , ", .{ idx, par.exists });
    }
    print("\n", .{});
}

const LTPS = struct {
    pts: []Pts,
    times: []u16,
};

/// A tracking represents a potentially invalid list of assignments between
/// objects across time. Objects have unique IDs which also serve as an index
/// into a slice of Pts. Each object may also have a Parent, which may in
/// fact be a mother-daughter cell relationship, but usually is just a way
/// of referring to the same cell at an earlier timepoint. The Parent may be
/// null, which means the cell is in the first frame of the image series, or has
/// appeared from the background or entered through the image boundaries.
const Tracking = struct {
    pts: []Pts,
    times: []u16,
    parentID: []ParentID,
    selfID: []u32,

    // pub fn assertValid(self: Tracking) bool {
    //     _ = self;
    // }
};

const PtId = struct { pt: Pts, id: u32 };
const PtIdTime = struct { pt: Pts, id: u32, time: u16 };
const PtIdTimeParent = struct { pt: Pts, id: u32, time: u16, parent_id: ?u32 };
const Tracking2D = []PtIdTimeParent;

fn mulFloor(a: anytype, b: f32) @TypeOf(a) {
    return @floatToInt(@TypeOf(a), @intToFloat(f32, a) * b);
}

// Get an SDL window we can use for visualizing algorithms.
const sdlw = @import("sdl-window.zig");
var win: sdlw.Window = undefined;

pub fn pt2screen(p: Pts) [2]i32 {
    const x = @floatToInt(i32, p[0] * 750 + 25);
    const y = @floatToInt(i32, p[1] * 750 + 25);
    return .{ x, y };
}

test "test generateTracking()" {
    try sdlw.initSDL();
    defer sdlw.quitSDL();
    win = try sdlw.Window.init(1000, 800);

    // var tracking = try generateTracking(allocator, .{});
    var tracking = try generateTrackingRandomWalk(allocator, 1000);
    defer allocator.free(tracking);

    try nearestParentAssignment(tracking);

    var temp__ = try allocator.alloc(PtId, 1_000);
    allocator.free(temp__);
    // print("{any}\n\n", .{p});
    // print("{any} \n", .{tracking});

    // win.markBounds();

    // sort by (time, x-coord)
    const lt = struct {
        fn lt(ctx: void, t0: PtIdTimeParent, t1: PtIdTimeParent) bool {
            _ = ctx;
            if (t0.id < t1.id) return true;
            return false;
        }
    }.lt;
    _ = lt;

    // std.sort.heap(PtIdTimeParent, tracking, {}, lt);
    // for (tracking) |p| {
    //     const x = @floatToInt(i32, p.pt[0] * 750 + 25);
    //     const y = @floatToInt(i32, p.pt[1] * 750 + 25);
    //     // const y = @floatToInt(i32, @intToFloat(f32, p.time) / 30 * 750 + 25);
    //     im.drawCircle([4]u8, win.pix, x, y, 3, .{ 255, 255, 255, 255 });

    //     if (p.parent_id) |parent_id| {
    //         const parent_node = tracking[parent_id];
    //         assert(parent_node.id == parent_id);

    //         const x2 = @floatToInt(i32, parent_node.pt[0] * 750 + 25);
    //         const y2 = @floatToInt(i32, parent_node.pt[1] * 750 + 25);
    //         // const y2 = @floatToInt(i32, @intToFloat(f32, parent_node.time) / 30 * 750 + 25);
    //         im.drawCircle([4]u8, win.pix, x2, y2, 3, .{ 0, 0, 255, 255 });
    //         im.drawLineInBounds([4]u8, win.pix, x, y, x2, y2, .{ 255, 255, 0, 255 });
    //         win.awaitKeyPressAndUpdateWindow();
    //     }
    // }
    // try win.update();
}

const TrackingParams = struct {
    n_times: u16 = 30,
    n_starting_cells: u16 = 40,
    n_div_upperbound: u16 = 3,
    n_enter_upperbound: u16 = 3,
    n_exit_upperbound: u16 = 3,
};

pub fn generateTracking(a2: Allocator, tp: TrackingParams) !Tracking2D {
    var arena = std.heap.ArenaAllocator.init(a2);
    defer arena.deinit();
    const a = arena.allocator();

    const n_times = tp.n_times;

    var n_divisions = try a.alloc(u16, n_times);
    for (n_divisions) |*v| v.* = random.uintAtMost(u16, tp.n_div_upperbound);

    var n_enter = try a.alloc(u16, n_times);
    for (n_enter) |*v| v.* = random.uintAtMost(u16, tp.n_enter_upperbound);

    var n_exit = try a.alloc(u16, n_times);
    for (n_exit) |*v| v.* = random.uintAtMost(u16, tp.n_enter_upperbound);

    var n_cells = try a.alloc(u16, n_times);
    for (n_cells, 0..) |*v, i| v.* = tp.n_starting_cells + n_divisions[i] + n_enter[i] - n_exit[i];

    const n_cell_cumsum = try cumsum(a, u16, n_cells);
    const n_total_cells = sum_u64(u16, n_cells);

    var pts = try a.alloc(Pts, n_total_cells);
    for (pts) |*v| v.* = .{ random.float(f32), random.float(f32) };

    var selfID = try a.alloc(u32, n_total_cells);
    for (selfID, 0..) |*v, i| v.* = @intCast(u32, i);

    var tracking = try a2.alloc(PtIdTimeParent, n_total_cells);
    for (tracking, 0..) |*v, i| {
        v.* = .{
            .pt = .{ random.float(f32), random.float(f32) },
            .id = @intCast(u32, i),
            .time = try binIdx(u16, n_cell_cumsum, @intCast(u16, i)),
            .parent_id = null,
        };
    }
    return tracking;
}

pub fn generateTrackingRandomWalk(a: Allocator, n_total_cells: u32) !Tracking2D {
    var tracking = try a.alloc(PtIdTimeParent, n_total_cells);

    var global_id: u32 = 0;
    var parent = PtIdTimeParent{
        .pt = .{ random.float(f32), random.float(f32) },
        .id = global_id,
        .time = random.uintLessThan(u16, 4),
        .parent_id = null,
    };
    tracking[global_id] = parent;
    global_id += 1;

    while (global_id < tracking.len) {
        // if beyond last frame, random exit or random entry, then start fresh

        var cell: PtIdTimeParent = undefined;

        if (random.float(f32) < 0.1) {
            cell = .{
                .pt = .{ random.float(f32), random.float(f32) },
                .id = global_id,
                .time = random.uintLessThan(u16, 4),
                .parent_id = null,
            };
        } else {
            cell = .{
                .pt = .{ parent.pt[0] + 0.05 * random.float(f32), parent.pt[1] + 0.05 * random.float(f32) },
                .id = global_id,
                .time = parent.time + 1,
                .parent_id = parent.id,
            };
        }

        tracking[global_id] = cell;
        parent = cell;
        global_id += 1;
    }

    return tracking;
}

fn cumsum(a: Allocator, comptime T: type, arr: []const T) ![]T {
    var tot = try a.alloc(T, arr.len + 1);
    tot[0] = 0;
    for (1..tot.len) |i| {
        tot[i] = tot[i - 1] + arr[i - 1];
    }
    return tot;
}

fn binIdx(comptime T: type, arr: []const T, val: T) !u16 {
    var idx: u16 = 0;
    while (idx < arr.len) : (idx += 1) {
        if (arr[idx] <= val and arr[idx + 1] >= val) return idx;
    }
    return error.OOB;
}

fn sum_u64(comptime T: type, arr: []const T) u64 {
    var total: u64 = 0;
    for (arr) |v| {
        total += v;
    }
    return total;
}

const ParentID = ?usize;

const nn_tools = @import("kdtree2d.zig");

fn ltPtsIdx(idx: u8, lhs: [2]f32, rhs: [2]f32) bool {
    return lhs[idx] < rhs[idx];
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

    for (a, 0..) |x, i| {
        for (b, 0..) |y, j| {
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
