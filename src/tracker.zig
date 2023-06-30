const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const assert = std.debug.assert;
const min = std.math.min;
const max = std.math.max;

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Local imports
const im = @import("image_base.zig");
const Tracer = @import("fn-tracer.zig").Tracer;
var tracer: Tracer(100) = undefined;
// Get an SDL window we can use for visualizing algorithms.
const sdlw = @import("sdl-window.zig");
var win: ?sdlw.Window = null;
const nn_tools = @import("kdtree2d.zig");

const Pt3D = [3]f32;
const Pt = [2]f32;

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

// const test_home = "/Users/broaddus/Desktop/work/isbi/zig-tracker/test-artifacts/track/";
// test {
//     // std.testing.refAllDecls(@This());
// }

fn distEuclid(comptime T: type, x: T, y: T) f32 {
    return switch (T) {
        Pt => (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]),
        Pt3D => (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) + (x[2] - y[2]) * (x[2] - y[2]),
        else => unreachable,
    };
}

// Runs assignment over each consecutive pair of pointclouds in time order.
pub fn trackOverFramePairs(tracking: Tracking2D) !void {

    // sort by (time, x-coord)
    // this is necessary for
    const lt = struct {
        fn lt(ctx: void, t0: TrackedCell, t1: TrackedCell) bool {
            _ = ctx;
            if (t0.time < t1.time) return true;
            if (t0.time == t1.time and t0.pt[0] < t1.pt[0]) return true;
            return false;
        }
    }.lt;

    std.sort.heap(TrackedCell, tracking.items, {}, lt);

    const timebounds = try tracking.getTimeboundsOfSorted(allocator);
    defer allocator.free(timebounds);

    // iterate over all frame pairs
    var t0_start: u32 = 0;
    for (0..timebounds.len - 1) |tb_idx| {
        const t0_end = timebounds[tb_idx].cum;
        const t1_end = timebounds[tb_idx + 1].cum;
        defer t0_start = t0_end;

        const trackslice_prev = tracking.items[t0_start..t0_end];
        const trackslice_curr = tracking.items[t0_end..t1_end];

        // Clear the screen. Draw prev pts in green, curr pts in blue.
        drawPts(trackslice_prev, .{ 0, 255, 0, 255 });
        drawPts(trackslice_curr, .{ 255, 0, 0, 255 });

        // try connectFramesGreedyDumb(trackslice_prev, trackslice_curr);
        try connectFramesGreedy(trackslice_prev, trackslice_curr);

        // Draw teal lines showing connections
        drawLinks(trackslice_curr, tracking, .{ 255, 255, 0, 255 });

        try connectFramesNearestNeib(trackslice_prev, trackslice_curr);

        // Draw red lines for connections
        drawLinks(trackslice_curr, tracking, .{ 0, 0, 255, 255 });
    }
}

pub fn drawPts(trackslice: []TrackedCell, color: [4]u8) void {
    if (win) |w| {
        for (w.pix.img) |*v| v.* = .{ 0, 0, 0, 255 };
        for (trackslice) |p| {
            const x = @floatToInt(i32, p.pt[0] * 750 + 25);
            const y = @floatToInt(i32, p.pt[1] * 750 + 25);
            // const y = @floatToInt(i32, @intToFloat(f32, p.time) / 30 * 750 + 25);
            im.drawCircle([4]u8, w.pix, x, y, 3, color);
        }
    }
}

pub fn drawLinks(trackslice: []TrackedCell, tracking: Tracking2D, color: [4]u8) void {
    if (win) |*w| {
        for (trackslice) |*p| {
            if (p.parent_id) |pid| {
                const x = @floatToInt(i32, p.pt[0] * 750 + 25);
                const y = @floatToInt(i32, p.pt[1] * 750 + 25);
                const parent = tracking.getID(pid).?;
                const x2 = @floatToInt(i32, parent.pt[0] * 750 + 25);
                const y2 = @floatToInt(i32, parent.pt[1] * 750 + 25);
                im.drawLineInBounds([4]u8, w.pix, x, y, x2, y2, color);
            }
        }
        w.awaitKeyPressAndUpdateWindow();
    }
}

// array order is [a,b]. i.e. a has stride nb. b has stride 1.
pub fn pairwiseDistances(al: Allocator, comptime T: type, a: []const T, b: []const T) ![]f32 {
    const na = a.len;
    const nb = b.len;

    var cost = try al.alloc(f32, na * nb);
    for (cost) |*v| v.* = 0;

    for (a, 0..) |x, i| {
        for (b, 0..) |y, j| {
            switch (T) {
                Pt => cost[i * nb + j] = distEuclid(Pt, x, y),
                TrackedCell => cost[i * nb + j] = distEuclid(Pt, x.pt, y.pt),
                else => unreachable,
            }
        }
    }

    return cost;
}

// Implementation of the Munkres Algorithm for optimal (minimal cost) linear
// sum assignment, but specialized on cell tracking where
// 1-2 assignment is possible. In fact 0-1, 1-0, 1-1, 1-2 assignments are all possible! They correspond to:
// 0-1 A cell enters the field of view through e.g. an image boundary, or appears from previously undetected state.
// 1-0 A cell dies or leaves through the image boundary.
// 1-1 A cell moves through time uneventfully (the most common case).
// 1-2 A cell divides into two daughters.
// We rule out 1-3 assignments as unrealistic for common framerates.
pub fn connectFramesMunkes(trackslice_prev: []const TrackedCell, trackslice_curr: []TrackedCell) !void {
    _ = trackslice_curr;
    _ = trackslice_prev;
}

// Alternative Greedy Linking where division costs are greater for the second child than
// for the first. This is done by first
pub fn connectFramesGreedy2(trackslice_prev: []const TrackedCell, trackslice_curr: []TrackedCell) !void {
    _ = trackslice_curr;
    _ = trackslice_prev;
}

// Create an array of all considered edges, perhaps throwing away unlikelies.
// Sort them based on cost(cell,cell).
// Assign the cheapest edges first. Keep track of the number of in/out edges for each cell.
// Don't assign edges that would violate constraints.
// You can go through the edges in a single pass from cheapest to most expensive, because once
// an edge becomes invalid it never becomes valid in the future. There is no backtracking!
pub fn connectFramesGreedyFaster(trackslice_prev: []const TrackedCell, trackslice_curr: []TrackedCell) !void {
    const tspan = tracer.start(@src().fn_name);
    defer tspan.stop();

    const na = trackslice_prev.len;
    const nb = trackslice_curr.len;

    // count number of out edges on A
    var n_out_parent = try allocator.alloc(u8, na);
    defer allocator.free(n_out_parent);
    for (n_out_parent) |*v| v.* = 0;

    // count number of in edges on B
    var n_in_child = try allocator.alloc(u8, nb);
    defer allocator.free(n_in_child);
    for (n_in_child) |*v| v.* = 0;

    // const cost = try pairwiseDistances(allocator, @TypeOf(va[0]), va, vb);
    // const CostEdgePair = struct { cost: f32, parent: TrackedCell, child: TrackedCell };
    const CostEdgePair = struct { cost: f32, idx_parent: usize, idx_child: usize };
    const edges = try allocator.alloc(CostEdgePair, na * nb);
    defer allocator.free(edges);

    for (trackslice_prev, 0..na) |p, i| {
        for (trackslice_curr, 0..nb) |c, j| {
            edges[i * nb + j] = .{ .cost = distEuclid(Pt, p.pt, c.pt), .idx_parent = i, .idx_child = j };
        }
    }

    const lt = struct {
        fn lt(context: void, a: CostEdgePair, b: CostEdgePair) bool {
            _ = context;
            return a.cost < b.cost;
        }
    }.lt;

    std.sort.heap(CostEdgePair, edges, {}, lt);

    // Iterate only one time over the edges in increasing order of cost.
    for (edges) |c| {

        // If the parent already has two children or the child is already assigned, skip this edge
        if (n_out_parent[c.idx_parent] == 2 or n_in_child[c.idx_child] == 1) continue;

        if (c.cost > 100) continue;

        // Make the assignment, then mark both the parent and child as assigned.
        // TODO: we don't really need n_in_child because we can just check trackslice[c.idx_child].parent_id == null
        trackslice_curr[c.idx_child].parent_id = trackslice_prev[c.idx_parent].id;
        n_out_parent[c.idx_parent] += 1;
        n_in_child[c.idx_child] += 1;
    }
}

// Put all edges between two frames into a PriorityQueue and add them to the solution greedily,
// without violating constraints.
pub fn connectFramesGreedy(trackslice_prev: []const TrackedCell, trackslice_curr: []TrackedCell) !void {
    const tspan = tracer.start(@src().fn_name);
    defer tspan.stop();

    const va = trackslice_prev;
    const vb = trackslice_curr;
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

    const cost = try pairwiseDistances(allocator, @TypeOf(va[0]), va, vb);
    defer allocator.free(cost);

    var asgn = try allocator.alloc(u8, na * nb);
    defer allocator.free(asgn);
    for (asgn) |*v| v.* = 0;

    const CostEdgePair = struct { cost: f32, ia: u32, ib: u32 };
    const lt = struct {
        fn lt(context: void, a: CostEdgePair, b: CostEdgePair) std.math.Order {
            _ = context;
            return std.math.order(a.cost, b.cost);
        }
    }.lt;

    // Place costs into priority queue
    var edgeQ = std.PriorityQueue(CostEdgePair, void, lt).init(allocator, {});
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

        if (aout[edge.ia] == 2 or bin[edge.ib] == 1) continue;
        asgn[edge.ia * nb + edge.ib] = 1;

        aout[edge.ia] += 1;
        bin[edge.ib] += 1;
    }

    // Assign parent id to current cells
    for (vb, 0..) |*current_cell, i| {
        current_cell.parent_id = null;
        for (va, 0..) |parent_cell, j| {
            // if the edge is active, then assign and break (can only have 1 parent max)
            if (asgn[j * nb + i] == 1) {
                assert(current_cell.parent_id == null); // assert only one possible parent!
                current_cell.parent_id = parent_cell.id;
            }
        }
    }
}

// Iterate over consecutive timepoints in Tracking and connect points to
// nearest parent in the previous frame. Ignore all constraints on the number
// of children a cell can have! Don't look at any other costs bust Euclidean distance.
pub fn connectFramesNearestNeib(trackslice_prev: []const TrackedCell, trackslice_curr: []TrackedCell) !void {
    const tspan = tracer.start(@src().fn_name);
    defer tspan.stop();
    // Make the connection using fast nearest neib lookup.
    for (trackslice_curr) |*p| {
        const parent_idx = nn_tools.findNearestNeibFromSortedListGeneric(TrackedCell, trackslice_prev, p.pt);
        const parent = trackslice_prev[parent_idx];
        p.parent_id = parent.id;
    }
}

// Assign cells to parents greedily, picking the best parents first, but iterating
// over cells in a random order.
pub fn connectFramesGreedyDumb(trackslice_prev: []const TrackedCell, trackslice_curr: []TrackedCell) !void {
    const tspan = tracer.start(@src().fn_name);
    defer tspan.stop();
    // keep track of the number of children map to a given parent
    var has_n_children = try allocator.alloc(u3, trackslice_prev.len);
    for (has_n_children) |*v| v.* = 0;
    defer allocator.free(has_n_children);

    // Find the optimal assignment
    for (trackslice_curr) |*p| {
        const Best = ?struct { parent: TrackedCell, score: f32, idx: usize };
        var best: Best = null;

        // iterate over all possible parents and find the best one (lowest score, still available)
        for (trackslice_prev, 0..) |p0, i| {
            const d = distEuclid(@TypeOf(p.pt), p.pt, p0.pt);
            // if the parent is not available, then skip.
            if (has_n_children[i] >= 2) continue;
            // if the parent is available and best is not null, but the distance isn't the best, then skip.
            if (best) |b| {
                if (d > b.score) continue;
            }
            // otherwise we update best.
            best = .{ .parent = p0, .score = d, .idx = i };
        }

        if (best) |b| {
            has_n_children[b.idx] += 1;
            p.parent_id = b.parent.id;
        }
    }
}

const TrackedCell = struct { pt: Pt, id: u32, time: u16, parent_id: ?u32 };

const Tracking2D = struct {
    items: []TrackedCell,

    const TimeCount = struct { time: u16, count: u16, cum: u32 };

    pub fn getTimeboundsOfSorted(this: @This(), a: Allocator) ![]TimeCount {
        const tracking = this.items;

        // sort by (time, x-coord)
        const lt = struct {
            fn lt(ctx: void, t0: TrackedCell, t1: TrackedCell) bool {
                _ = ctx;
                if (t0.time < t1.time) return true;
                // if (t0.time == t1.time and t0.pt[0] < t1.pt[0]) return true;
                return false;
            }
        }.lt;

        assert(std.sort.isSorted(TrackedCell, tracking, {}, lt));

        // std.sort.heap(TrackedCell, tracking, {}, lt);

        // then find time boundaries
        // Can't be longer than this. times are discrete and >= 0.
        var timebounds = try a.alloc(TimeCount, tracking[tracking.len - 1].time + 1);
        // defer allocator.free(timebounds);

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

        return timebounds;
    }

    pub fn getID(this: @This(), id: u32) ?TrackedCell {
        for (this.items) |t| {
            if (t.id == id) return t;
        }
        return null;
    }

    // TODO: getIDfromSorted() // bisection search
    // TODO: getIDfromCompactSorted() // const time direct access
};

pub fn pt2screen(p: Pt) [2]i32 {
    const x = @floatToInt(i32, p[0] * 750 + 25);
    const y = @floatToInt(i32, p[1] * 750 + 25);
    return .{ x, y };
}

pub fn main() !void {
    try sdlw.initSDL();
    defer sdlw.quitSDL();
    // win = try sdlw.Window.init(1000, 800);

    tracer = try Tracer(100).init();

    var tracking = try generateTrackingLineage(allocator, 1000);
    defer allocator.free(tracking.items);

    try trackOverFramePairs(tracking);

    try tracer.analyze(allocator);
}

// Generate a simulated lineage with cells, divisions and apoptosis.
pub fn generateTrackingLineage(a: Allocator, n_total_cells: u32) !Tracking2D {
    var tracking = try std.ArrayList(TrackedCell).initCapacity(a, n_total_cells);
    var unfinished_lineage_q = try std.ArrayList(TrackedCell).initCapacity(a, 100);
    const jumpdist: f32 = 0.1;

    // cumulative distribution of events
    const p_new_lineage: f32 = 0.00;
    const p_continue: f32 = 0.9 + p_new_lineage;
    const p_divide: f32 = 0.075 + p_continue;
    const p_death: f32 = 0.025 + p_divide;
    _ = p_death;

    // Add ten cells to the starting frame
    for (0..10) |_| {
        const cell = .{
            .pt = .{ random.float(f32), random.float(f32) },
            .id = @intCast(u32, tracking.items.len),
            .time = 0, //random.uintLessThan(u16, 4),
            .parent_id = null,
        };
        unfinished_lineage_q.appendAssumeCapacity(cell);
        tracking.appendAssumeCapacity(cell);
    }

    // assert(p_death == 1.0);

    // Keep adding cells to the lineage tree until we've exhausted the capacity
    while (tracking.items.len < tracking.capacity) {

        // Randomize the order of unfinished lineages so each time we randomly sample from the queue
        random.shuffle(TrackedCell, unfinished_lineage_q.items);

        const rchoice = random.float(f32);
        if (rchoice < p_new_lineage or unfinished_lineage_q.items.len == 0) {
            // 10% chance that we start a new trajectory (somewhere in first 4 frames)

            const cell = .{
                .pt = .{ random.float(f32), random.float(f32) },
                .id = @intCast(u32, tracking.items.len),
                .time = 0, //random.uintLessThan(u16, 4),
                .parent_id = null,
            };
            unfinished_lineage_q.appendAssumeCapacity(cell);
            tracking.appendAssumeCapacity(cell);
        } else if (rchoice < p_continue) {
            // 80% chance we continue the parent lineage

            const dx = jumpdist * (0.5 - random.float(f32));
            const dy = jumpdist * (0.5 - random.float(f32));

            // parent = if (unfinished_lineage_q.popOrNull()) |p| p else continue;

            const parent = unfinished_lineage_q.pop();
            const cell = .{
                .pt = .{ parent.pt[0] + dx, parent.pt[1] + dy },
                .id = @intCast(u32, tracking.items.len),
                .time = parent.time + 1,
                .parent_id = parent.id,
            };
            unfinished_lineage_q.appendAssumeCapacity(cell);
            tracking.appendAssumeCapacity(cell);
        } else if (rchoice < p_divide) {
            // 5% chance we divide. pop once, enqueue twice.
            // Daughter cells appear on opposite sides of the mother, equally spaced

            const parent = unfinished_lineage_q.pop();
            const dx = jumpdist * (0.5 - random.float(f32));
            const dy = jumpdist * (0.5 - random.float(f32));

            // create and enqueue cell 1
            const cell1 = .{
                .pt = .{ parent.pt[0] + dx, parent.pt[1] + dy },
                .id = @intCast(u32, tracking.items.len),
                .time = parent.time + 1,
                .parent_id = parent.id,
            };
            unfinished_lineage_q.appendAssumeCapacity(cell1);
            tracking.appendAssumeCapacity(cell1);

            if (tracking.items.len == tracking.capacity) break;

            // create and enqueue cell 2
            const cell2 = .{
                .pt = .{ parent.pt[0] - dx, parent.pt[1] - dy },
                .id = @intCast(u32, tracking.items.len),
                .time = parent.time + 1,
                .parent_id = parent.id,
            };
            unfinished_lineage_q.appendAssumeCapacity(cell2);
            tracking.appendAssumeCapacity(cell2);
        } else {
            // 5% chance the lineage dies

            _ = unfinished_lineage_q.pop();
        }
    }

    // This fails to compile because calling free on this slice later
    return .{ .items = tracking.items };
}
