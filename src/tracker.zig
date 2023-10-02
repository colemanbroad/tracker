const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const assert = std.debug.assert;
const min = std.math.min;
const max = std.math.max;

var prng = std.rand.DefaultPrng.init(1);
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
var win_plot: ?sdlw.Window = null;
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

// const test_home = "/Users/broaddus/work/isbi/zig-tracker/test-artifacts/track/";
// test {
//     // std.testing.refAllDecls(@This());
// }

// This function takes the path of an LBEP file which has Labels, Begin, End, and Parent.
// It's a way of encoding a tracking.
// We need the centerpoints associated with each object in the tracking to really work with
// this data. Maybe we can do the initial loading and parsing with existing python tools,
// and then we can write it in a format similar to our existing Tracking...

// A Tracking is a slice of TrackedCell, which has a point, id, time, and parent_id...
// We can make a data structure that has all of this information and pass it from python.
// Or save it to disk.

// const TrackedCell = struct { pt: Pt, id: u32, time: u16, parent_id: ?u32 };

//
// const TrackedCellJson = struct { time: u32, id: i32, pt: [2]u16, parent_id: [2]i32, isbi_id: i32 };
const TrackedCellJson = struct { time: u32, id: i32, pt: Pt, parent_id: [2]i32, isbi_id: i32 };

fn getTrackedCells(al: Allocator) !std.json.Parsed([]TrackedCellJson) {
    const s = @embedFile("tracked_cells.json");
    const parsedData = try std.json.parseFromSlice([]TrackedCellJson, al, s[0..], .{});
    return parsedData;
    // defer parsedData.deinit();
}

// fn load_isbi_data(dir: []const u8) !void {
//     _ = dir;
// }

// Robert Sedgewick's non-recursive permutation algorithm
// https://sedgewick.io/wp-content/uploads/2022/03/2002PermGeneration.pdf
//   from https://en.wikipedia.org/wiki/Heap%27s_algorithm?useskin=vector
// The implementation derives directly from the recursive one, but uses an explicit
// stack counter (c) instead of an implicit one. The idea is that we can recursively
// define a permutation p(n) as n * p(n-1) i.e. p(x) = for i=0..x.len {swap(0,i); perm()}
//
fn enumeratePermutations(arr: []u32) !void {
    var count: u32 = 0;

    const n = arr.len;
    var c = try allocator.alloc(u32, n);
    for (c) |*x| x.* = 0;

    print("{d:10} == {d} \n", .{ count, arr });
    count += 1;

    var i: u32 = 0;
    // i is the index into c. arr.len = c.len = n.
    while (i < n) {
        if (c[i] < i) {
            if (i % 2 == 0) {
                // swap 0 , i
                const tmp = arr[0];
                arr[0] = arr[i];
                arr[i] = tmp;
            } else {
                // swap c[i] , i
                const tmp = arr[c[i]];
                arr[c[i]] = arr[i];
                arr[i] = tmp;
            }

            print("{d:10} == {d} \n", .{ count, arr });
            count += 1;

            c[i] += 1;
            i = 1;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
}

test "test enumeratePermutations" {
    print("\n", .{});
    const n: u32 = 10;
    var arr = try allocator.alloc(u32, n);
    for (arr, 0..) |*a, i| a.* = @as(u32, @intCast(i));

    try enumeratePermutations(arr);
}

fn distEuclid(comptime T: type, x: T, y: T) f32 {
    return switch (T) {
        Pt => (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]),
        Pt3D => (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) + (x[2] - y[2]) * (x[2] - y[2]),
        else => unreachable,
    };
}

pub const cc = struct {
    pub usingnamespace @cImport({
        @cInclude("SDL2/SDL.h");
    });
};

const blue = .{ 255, 0, 0, 255 };
const grey = .{ 128, 128, 128, 255 };
const green = .{ 0, 255, 0, 255 };
const red = .{ 0, 0, 255, 255 };
const black = .{ 0, 0, 0, 255 };
const white = .{ 255, 255, 255, 255 };
const gold = .{ 0, 215, 255, 255 };
const pink = .{ 193, 182, 255, 255 };

// Runs assignment over each consecutive pair of pointclouds in time order.
pub fn trackOverFramePairs(orig_tracking: Tracking) !void {

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

    std.sort.heap(TrackedCell, orig_tracking.items, {}, lt);

    // make a copy so we can keep multiple solutions alive at the same time and draw them
    // on top of each other.
    const tracking = Tracking{
        .items = try allocator.alloc(TrackedCell, orig_tracking.items.len),
    };
    // TrackedCell is a copyable struct.
    for (orig_tracking.items, 0..) |x, i| tracking.items[i] = x;

    var timebounds = try tracking.getTimeboundsOfSorted(allocator);
    defer timebounds.deinit();

    // iterate over all frame pairs
    var tb_idx: u16 = 0;

    while (true) {
        const tb_zero = if (timebounds.get(tb_idx + 0)) |x| x else break;
        const tb_one = if (timebounds.get(tb_idx + 1)) |x| x else break;
        // const tb_two = if (timebounds.get(tb_idx + 2)) |x| x else break;

        // print("lengths zero {} one {} two {} \n", .{ trackslice_zero.len, trackslice_one.len, trackslice_two.len });
        // print("Timebounds zero {[start]} {[stop]}\n", tb_zero);
        // print("Timebounds one {[start]} {[stop]}\n\n", tb_one);
        // print("Timebounds two {[start]} {[stop]}\n\n", tb_two);

        // Clear the screen and draw circes at each cell.
        for (win.?.pix.img) |*v| v.* = .{ 0, 0, 0, 255 };

        // const orig_trackslice_zero = orig_tracking.items[tb_zero.start..tb_zero.stop];
        // _ = orig_trackslice_zero;
        // const orig_trackslice_one = orig_tracking.items[tb_one.start..tb_one.stop];
        // const orig_trackslice_two = orig_tracking.items[tb_two.start..tb_two.stop];

        // // try linkFramesGreedy(trackslice_zero, trackslice_one);
        // // try linkFramesGreedy(trackslice_one, trackslice_two);
        // drawLinksToParent(orig_trackslice_one, orig_tracking, red);
        // drawLinksToParent(orig_trackslice_two, orig_tracking, red);

        const trackslice_zero = tracking.items[tb_zero.start..tb_zero.stop];
        const trackslice_one = tracking.items[tb_one.start..tb_one.stop];
        // const trackslice_two = tracking.items[tb_two.start..tb_two.stop];

        // try linkFramesGreedyDumb(trackslice_zero, trackslice_one);
        // try linkFramesGreedyDumb(trackslice_one, trackslice_two);
        try linkFramesMunkes(trackslice_zero, trackslice_one);
        // try linkFramesMunkes(trackslice_one, trackslice_two);
        // try linkFramesGreedyDumb(trackslice_zero, trackslice_one);

        drawPts(trackslice_zero, blue);
        drawPts(trackslice_one, green);
        // drawPts(trackslice_two, green);
        drawLinksToParent(trackslice_one, tracking, red);
        // drawLinksToParent(trackslice_two, tracking, red);

        if (win) |*w| {
            w.update() catch unreachable;
            const key = w.awaitKeyPress();
            switch (key) {
                cc.SDLK_h => tb_idx -|= 1,
                cc.SDLK_l => tb_idx +|= 1,
                else => {},
            }
        } else {
            tb_idx += 1;
        }
    }
}

pub fn drawPts(trackslice: []TrackedCell, color: [4]u8) void {
    if (win) |w| {
        for (trackslice) |p| {
            const p0 = pt2screen(p.pt);
            im.drawCircle([4]u8, w.pix, p0[0], p0[1], 3, color);
        }
    }
}

pub fn drawLinksToParent(trackslice: []TrackedCell, tracking: Tracking, color: [4]u8) void {
    if (win) |*w| {
        for (trackslice) |*p| {
            if (p.parent_id) |pid| {
                const p0 = pt2screen(p.pt);
                const parent = tracking.getCellFromID(pid).?;
                const p1 = pt2screen(parent.pt);
                im.drawLineInBounds([4]u8, w.pix, p0[0], p0[1], p1[0], p1[1], color);
            }
        }
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

// Implementation of the Munkres Algorithm for optimal (minimal cost) linear-sum assignmnet,
// but specialized for cell tracking where 1-2 assignment is possible.
// In fact 0-1, 1-0, 1-1, 1-2 assignments are all possible! They correspond to:
// 0-1 A cell enters the field of view through e.g. an image boundary, or appears from previously undetected state.
// 1-0 A cell dies or leaves through the image boundary.
// 1-1 A cell moves through time uneventfully (the most common case).
// 1-2 A cell divides into two daughters.
// We rule out 1-3 assignments as unrealistic for common framerates.
// Implementation from [duke](https://users.cs.duke.edu/~brd/Teaching/Bio/asmb/current/Handouts/munkres.html)

// From [duke](https://users.cs.duke.edu/~brd/Teaching/Bio/asmb/current/Handouts/munkres.html)
// As each assignment is chosen that row and column are eliminated from
// consideration.  The question is raised as to whether there is a better
// algorithm.  In fact there exists a polynomial runtime complexity algorithm
// for solving the assignment problem developed by James Munkre's in the late
// 1950's despite the fact that some references still describe this as a problem
// of exponential complexity.

// The following 6-step algorithm is a modified form of the original Munkres'
// Assignment Algorithm (sometimes referred to as the Hungarian Algorithm). This
// algorithm describes to the manual manipulation of a two-dimensional matrix by
// starring and priming zeros and by covering and uncovering rows and columns.
// This is because, at the time of publication (1957), few people had access to
// a computer and the algorithm was exercised by hand.
const LinkState = enum { none, starred, primed };
const RowColState = enum { noncovered, covered };

// n0
// n1
// link_costs
// min_cost_prev
// min_cost_curr
// link_state
// vert_cover_prev
// vert_cover_curr
// vert_star_prev
// vert_star_curr

// Covered row/col is gold. Noncovered is pink.
// Starred/Primed/Neither Zeros are Blue/Green/White
// Nonzero links are Grey
pub fn drawMatrix(src: std.builtin.SourceLocation, allstate: anytype) void {
    if (win_plot == null) return;
    print("source line: {any} \n", .{src.line});
    const a = allstate;
    for (win_plot.?.pix.img) |*v| v.* = .{ 0, 0, 0, 255 };

    const r = 3;

    for (0..a.n0) |j0| {
        const x0 = 150 + @as(i32, @intCast(j0 * 20));
        const y0 = 100;

        const c = a.vert_cover_prev[j0];
        const color: [4]u8 = switch (c) {
            .covered => pink,
            .noncovered => blue,
        };
        im.drawCircle([4]u8, win_plot.?.pix, x0, y0, r, color);
    }

    for (0..a.n1) |j1| {
        const x0 = 100;
        const y0 = 150 + @as(i32, @intCast(j1 * 20));

        const c = a.vert_cover_curr[j1];
        const color: [4]u8 = switch (c) {
            .covered => pink,
            .noncovered => blue,
        };
        im.drawCircle([4]u8, win_plot.?.pix, x0, y0, r, color);
    }

    for (0..a.n0) |j0| {
        for (0..a.n1) |j1| {
            const x0 = 150 + @as(i32, @intCast(j0 * 20));
            const y0 = 150 + @as(i32, @intCast(j1 * 20));

            const c = a.link_state[j0 * a.n1 + j1];
            var color: [4]u8 = switch (c) {
                .none => white,
                .starred => gold,
                .primed => red,
            };
            const cost = a.link_costs[j0 * a.n1 + j1];
            if (cost != 0.0) {
                color = grey;
            }
            im.drawCircle([4]u8, win_plot.?.pix, x0, y0, r, color);

            // const t0 = a.vert_star_prev[j0] == j1;
            // const t1 = a.vert_star_curr[j1] == j0;
            // if (t0 and t1) {
            //     im.drawCircleOutline(win_plot.?.pix, x0, y0, r + 3, gold);
            //     im.drawCircleOutline(win_plot.?.pix, x0, y0, r + 2, gold);
            //     im.drawCircleOutline(win_plot.?.pix, x0, y0, r + 1, gold);
            // } else if (t0) {
            //     im.drawCircleOutline(win_plot.?.pix, x0, y0, r + 3, pink);
            //     im.drawCircleOutline(win_plot.?.pix, x0, y0, r + 2, pink);
            //     im.drawCircleOutline(win_plot.?.pix, x0, y0, r + 1, pink);
            // } else if (t1) {
            //     im.drawCircleOutline(win_plot.?.pix, x0, y0, r + 3, red);
            //     im.drawCircleOutline(win_plot.?.pix, x0, y0, r + 2, red);
            //     im.drawCircleOutline(win_plot.?.pix, x0, y0, r + 1, red);
            // }
        }
    }

    win_plot.?.update() catch unreachable;
    // _ = win_plot.?.awaitKeyPress();
    // if (a.loop_count.* > 100) _ = win_plot.?.awaitKeyPress();
}

const Matrix = im.Img2D;

export fn pymunkres(link_costs: [*]f32, n0: u32, n1: u32, assignments: [*]u8) void {
    const link_matrix = Matrix(f32){
        .img = link_costs[0..(n0 * n1)],
        .nx = n0,
        .ny = n1,
    };

    const assignments_ = munkresAssignments(allocator, link_matrix) catch unreachable;
    for (assignments_.img, 0..) |x, i| {
        assignments[i] = x;
    }
}

pub fn munkresAssignments(allo: Allocator, link_costs_copy: Matrix(f32)) !Matrix(u8) {

    // First we have to make an nxn grid of edge costs
    // Then an nxn grid of algorithm state for each edge

    var _arena = std.heap.ArenaAllocator.init(allo);
    defer _arena.deinit();
    const aa = _arena.allocator();

    const n0 = link_costs_copy.nx;
    const n1 = link_costs_copy.ny;
    // const k = @min()

    var link_costs = try aa.alloc(f32, n0 * n1);
    var min_cost_prev = try aa.alloc(f32, n0);
    var min_cost_curr = try aa.alloc(f32, n1);
    var link_state = try aa.alloc(LinkState, n0 * n1);
    var assignment_solution = Matrix(u8){ .nx = n0, .ny = n1, .img = try allo.alloc(u8, n0 * n1) };
    var vert_cover_prev = try aa.alloc(RowColState, n0);
    var vert_cover_curr = try aa.alloc(RowColState, n1);
    var zero_list = std.AutoArrayHashMap([2]usize, void).init(aa);
    var loop_count = @as(u64, 0);
    var step5_series = try std.ArrayList([2]usize).initCapacity(aa, 3 * n1);

    const allstate = .{
        .n0 = n0,
        .n1 = n1,
        .link_costs = link_costs,
        .min_cost_prev = min_cost_prev,
        .min_cost_curr = min_cost_curr,
        .link_state = link_state,
        .vert_cover_prev = vert_cover_prev,
        .vert_cover_curr = vert_cover_curr,
        .zero_list = zero_list,
        .loop_count = &loop_count,
    };
    _ = allstate;

    // Generate costs. The optimal solution minimizes the sum of costs across
    // all possible assignments.
    for (link_costs, 0..) |*l, i| l.* = link_costs_copy.img[i];

    // Find min cost for rows and columns. Initialize with largest possible val
    for (min_cost_prev) |*c| c.* = 1000;
    for (min_cost_curr) |*c| c.* = 1000;
    for (0..n0) |j0| {
        for (0..n1) |j1| {
            const l = link_costs[j0 * n1 + j1];
            if (l < min_cost_prev[j0]) min_cost_prev[j0] = l;
            if (l < min_cost_curr[j1]) min_cost_curr[j1] = l;
        }
    }

    // Subtract the min across `prev` from each `curr`
    for (0..n0) |j0| {
        for (0..n1) |j1| {
            link_costs[j0 * n1 + j1] -= min_cost_prev[j0];
            if (link_costs[j0 * n1 + j1] == 0.0) {
                try zero_list.put(.{ j0, j1 }, {});
            }
        }
    }

    // Vertices start off noncovered. Links start off "none", i.e. unstarred and unprimed.
    for (link_state) |*v| v.* = .none;
    for (vert_cover_prev) |*v| v.* = .noncovered;
    for (vert_cover_curr) |*v| v.* = .noncovered;

    // Step 2 : Find a zero (Z) in the resulting matrix.  If there is no starred zero
    // in its row or column, star Z. Repeat for each element in the matrix.
    // Go to Step 3.
    // WARN: so the starred zeros depend on the order we traverse matrix. why doesn't this matter?
    // do step 2. greedily search through matrix elements and star them.
    for (0..n0) |j0| {
        for (0..n1) |j1| {
            const c = link_costs[j0 * n1 + j1];
            if (c != 0.0) continue;

            // if there is no starred zero in it's row or column, star Z
            const j1_maybe = idxOfZeroInCol(link_state, .starred, n0, n1, j0);
            if (j1_maybe != -1) continue;
            const j0_maybe = idxOfZeroInRow(link_state, .starred, n0, n1, j1);
            if (j0_maybe != -1) continue;

            // we've found an unstarred zero. star it!
            link_state[j0 * n1 + j1] = .starred;
        }
    }

    // drawMatrix(@src(), allstate);

    // Loop beginning with Step 3
    while (true) {
        loop_count += 1;

        // Step 3:  Cover each column containing a starred zero.  If K columns are
        // covered, the starred zeros describe a complete set of unique assignments.  In
        // this case, Go to DONE, otherwise, Go to Step 4.
        var n_covered = @as(u32, 0);
        for (0..n0) |j0| {
            const j1_maybe = idxOfZeroInCol(link_state, .starred, n0, n1, j0);
            if (j1_maybe == -1) continue;
            vert_cover_prev[j0] = .covered;
            n_covered += 1;
        }
        if (n_covered == vert_cover_prev.len) {
            print("We have a winner!\n", .{});
            // drawMatrix(@src(), allstate);
            for (link_state, 0..) |l, i| {
                assignment_solution.img[i] = if (l == .starred) 1 else 0;
            }
            return assignment_solution;
        }

        var uncovered_primed_zero = @as([2]usize, .{ 99, 99 });
        step4and6: while (true) {
            var iter = zero_list.iterator();

            // Step 4: Find a noncovered zero and prime it. If there is no starred zero in
            // the row containing this primed zero, Go to Step 5.
            // Otherwise, cover this row and uncover the column containing the starred zero.
            // Continue in this manner until there are no uncovered zeros left. Save the smallest
            // uncovered value and Go to Step 6.
            while (iter.next()) |j0j1| {
                const j0 = j0j1.key_ptr[0];
                const j1 = j0j1.key_ptr[1];
                // print("j0 = {} j1 = {}\n", .{ j0, j1 });

                // Find a noncovered zero and prime it. // blue+non-grey -> blue+red
                if (vert_cover_prev[j0] != .noncovered) continue;
                if (vert_cover_curr[j1] != .noncovered) continue;
                link_state[j0 * n1 + j1] = .primed;

                // If there is no starred zero in the row containing this primed zero, Go to Step 5.
                const j0_maybe = idxOfZeroInRow(link_state, .starred, n0, n1, j1);
                if (j0_maybe == -1) {
                    uncovered_primed_zero = .{ j0, j1 };
                    // print("j0 = {} j1 = {}\n", .{ j0, j1 });
                    // drawMatrix(@src(), allstate);
                    //             _ = win_plot.?.awaitKeyPress();
                    break :step4and6;
                }

                // Otherwise, cover this row and uncover the column containing the starred zero.
                vert_cover_curr[j1] = .covered;
                vert_cover_prev[@intCast(j0_maybe)] = .noncovered;
            }
            // Save the smallest uncovered value and go to step 6
            var smallest_uncovered_val = @as(f32, 1000);
            var smallest_uncovered_idx = @as([2]usize, .{ 99, 99 });
            for (0..n0) |j0| {
                for (0..n1) |j1| {
                    const c = link_costs[j0 * n1 + j1];
                    if (vert_cover_prev[j0] == .covered) continue;
                    if (vert_cover_curr[j1] == .covered) continue;
                    if (c > smallest_uncovered_val) continue;
                    smallest_uncovered_val = c;
                    smallest_uncovered_idx = .{ j0, j1 };
                }
            }

            // Step 6: Add the value found in Step 4 to every element of each covered row, and subtract it from every element of
            // each uncovered column.  Return to Step 4 without altering any stars, primes, or covered lines.
            for (0..n0) |j0| {
                for (0..n1) |j1| {
                    // Add to every covered row
                    if (vert_cover_curr[j1] == .covered) {
                        // BUG: sometimes assert fails. why?
                        // if (link_costs[j0 * n1 + j1] == 0.0) assert(zero_list.swapRemove(.{ j0, j1 }));
                        if (link_costs[j0 * n1 + j1] == 0.0) _ = zero_list.swapRemove(.{ j0, j1 });
                        link_costs[j0 * n1 + j1] += smallest_uncovered_val;
                    }
                    // Subtract from every covered column
                    if (vert_cover_prev[j0] == .noncovered) {
                        link_costs[j0 * n1 + j1] -= smallest_uncovered_val;
                        if (link_costs[j0 * n1 + j1] == 0.0) try zero_list.put(.{ j0, j1 }, {});
                    }
                }
            }
        }

        // Step 5: Construct a series of alternating primed and starred zeros as
        // follows. Let Z0 represent the uncovered primed zero found in Step 4. Let Z1
        // denote the starred zero in the column of Z0 (if any). Let Z2 denote the
        // primed zero in the row of Z1 (there will always be one). Continue until the
        // series terminates at a primed zero that has no starred zero in its column.
        // Unstar each starred zero of the series, star each primed zero of the series,
        // erase all primes and uncover every line in the matrix. Return to Step 3.

        // Let Z0 represent the uncovered primed zero found in Step 4.
        {
            var j0 = uncovered_primed_zero[0];
            var j1 = uncovered_primed_zero[1];

            step5_series.clearRetainingCapacity();

            var loop_idx = @as(u32, 0);
            while (true) {
                loop_idx += 1;
                if (loop_idx > 0) {
                    // drawMatrix(@src(), allstate);
                    // print("j0 = {} j1 = {}\n", .{ j0, j1 });
                }

                // Z0
                try step5_series.append(.{ j0, j1 });
                // Z1
                const j1_maybe = idxOfZeroInCol(link_state, .starred, n0, n1, j0);
                // Let Z1 denote the starred zero in the column of Z0 (if any).
                // Continue until the series terminates at a primed zero that has no starred zero in its column.
                if (j1_maybe == -1) {
                    break;
                }
                j1 = @intCast(j1_maybe);
                try step5_series.append(.{ j0, j1 });
                // Let Z2 denote the primed zero in the row of Z1 (there will always be one).
                j0 = @intCast(idxOfZeroInRow(link_state, .primed, n0, n1, j1));
            }
        }

        // drawMatrix(@src(), allstate);
        // swap primes and stars
        for (step5_series.items, 0..) |j0j1, idx| {
            const j0 = j0j1[0];
            const j1 = j0j1[1];
            if (idx % 2 == 0) {
                link_state[j0 * n1 + j1] = .starred;
            } else {
                link_state[j0 * n1 + j1] = .primed;
            }
        }

        // drawMatrix(@src(), allstate);

        // erase all primes and uncover every line in the matrix. Return to Step 3.
        for (link_state) |*ls| {
            if (ls.* == .primed) ls.* = .none;
        }
        for (vert_cover_prev) |*v| v.* = .noncovered;
        for (vert_cover_curr) |*v| v.* = .noncovered;

        // drawMatrix(@src(), allstate);
    }
}

pub fn linkFramesMunkes(trackslice_prev: []const TrackedCell, trackslice_curr: []TrackedCell) !void {

    // First we have to make an nxn grid of edge costs
    // Then an nxn grid of algorithm state for each edge

    const n0 = trackslice_prev.len;
    const n1 = trackslice_curr.len;

    var link_costs = Matrix(f32){
        .img = try allocator.alloc(f32, n0 * n1),
        .nx = @intCast(n0),
        .ny = @intCast(n1),
    };

    // Generate costs. The optimal solution minimizes the sum of costs across
    // all possible assignments.
    for (trackslice_prev, 0..) |v0, j0| {
        for (trackslice_curr, 0..) |v1, j1| {
            const c = distEuclid(Pt, v0.pt, v1.pt);
            link_costs.set(j0, j1, c * c);
        }
    }

    const assignment_solution = try munkresAssignments(allocator, link_costs);
    _ = assignment_solution;
    // print("{any}", .{assignment_solution});
}

fn idxOfZeroInRow(link_state: []const LinkState, state: LinkState, n0: usize, n1: usize, j1: usize) i32 {
    for (0..n0) |j0| {
        if (link_state[j0 * n1 + j1] == state) return @intCast(j0);
    }
    return -1;
}

fn idxOfZeroInCol(link_state: []const LinkState, state: LinkState, n0: usize, n1: usize, j0: usize) i32 {
    _ = n0;
    for (0..n1) |j1| {
        if (link_state[j0 * n1 + j1] == state) return @intCast(j1);
    }
    return -1;
}

pub fn printMunkresState(n0: usize, n1: usize, link_state: []const LinkState) void {
    _ = n1;
    for (link_state, 1..) |l, i| {
        const cab = switch (l) {
            .none => "n",
            .starred => "*",
            .primed => "\'",
        };
        print("{s}", .{cab});

        if (i % n0 == 0) {
            print("\n", .{});
        }
    }
    // print("\x1B[F\x1B[F\x1B[F\x1B[F\x1B[F\x1B[F\x1B[F\x1B[F\x1B[F\x1B[F", .{});
}

// test "test the munkres tracker" {}

// Alternative Greedy Linking where division costs are greater for the second child than
// for the first. This is done by first
pub fn linkFramesGreedy2(trackslice_prev: []const TrackedCell, trackslice_curr: []TrackedCell) !void {
    _ = trackslice_curr;
    _ = trackslice_prev;
}

// Create an array of all considered edges, perhaps throwing away unlikelies.
// Sort them based on cost(cell,cell).
// Assign the cheapest edges first. Keep track of the number of in/out edges for each cell.
// Don't assign edges that would violate constraints.
// You can go through the edges in a single pass from cheapest to most expensive, because once
// an edge becomes invalid it never becomes valid in the future. There is no backtracking!
pub fn linkFramesGreedyFaster(trackslice_prev: []const TrackedCell, trackslice_curr: []TrackedCell) !void {
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
pub fn linkFramesGreedy(trackslice_prev: []const TrackedCell, trackslice_curr: []TrackedCell) !void {
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
        try edgeQ.add(.{ .cost = c, .ia = @as(u32, @intCast(ia)), .ib = @as(u32, @intCast(ib)) });
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
pub fn linkFramesNearestNeib(trackslice_prev: []const TrackedCell, trackslice_curr: []TrackedCell) !void {
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
// over cells without sorting them first.
pub fn linkFramesGreedyDumb(trackslice_prev: []const TrackedCell, trackslice_curr: []TrackedCell) !void {
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

// This ID type aligns with data coming from python world
const CellID = struct { u16, u32 }; // Time, ID
const TrackedCell = struct { pt: Pt, id: u32, time: u16, parent_id: ?u32 };
// const TrackedCell = struct { pt: Pt, id: CellID, time: u16, parent_id: ?CellID };

const Tracking = struct {
    items: []TrackedCell,

    // const TimeCount = struct { time: u16, count: u16, cum: u32, start_id: u32, stopd_id: u32 };
    const TimeBound = struct { start: u32, stop: u32 };

    // pub fn getTimeboundsOfSorted(this: @This(), a: Allocator) ![]TimeCount {
    pub fn getTimeboundsOfSorted(this: @This(), a: Allocator) !std.AutoArrayHashMap(u16, TimeBound) {
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

        var count_cells_per_time = std.AutoArrayHashMap(u16, u32).init(a);
        for (tracking) |cell| {
            const val = if (count_cells_per_time.get(cell.time)) |c| c else 0;
            try count_cells_per_time.put(cell.time, val + 1);
        }
        defer count_cells_per_time.deinit();

        var idx_start: u32 = 0;
        var time_to_idx = std.AutoArrayHashMap(u16, TimeBound).init(a);
        var it = count_cells_per_time.iterator();
        while (it.next()) |kv| {
            const idx_stop = idx_start + kv.value_ptr.*;
            try time_to_idx.put(kv.key_ptr.*, .{ .start = idx_start, .stop = idx_stop });
            idx_start = idx_stop;
        }
        return time_to_idx;
    }

    pub fn getCellFromID(this: @This(), id: u32) ?TrackedCell {
        for (this.items) |t| {
            if (t.id == id) return t;
        }
        return null;
    }

    // TODO: getIDfromSorted() // bisection search
    // TODO: getIDfromCompactSorted() // const time direct access
};

pub fn pt2screen(p: Pt) [2]i32 {
    const x = @as(i32, @intFromFloat(p[0] * 750 + 25));
    const y = @as(i32, @intFromFloat(p[1] * 750 + 25));
    return .{ x, y };
}

pub fn main() !void {
    try sdlw.initSDL();
    defer sdlw.quitSDL();
    win = try sdlw.Window.init(1000, 800);
    win_plot = try sdlw.Window.init(500, 500);
    _ = win_plot.?.awaitKeyPress();

    tracer = try Tracer(100).init();

    var tracking = try generateTrackingLineage(allocator, 10_000);
    defer allocator.free(tracking.items);

    try trackOverFramePairs(tracking);

    try tracer.analyze(allocator);
}

// Generate a simulated lineage with cells, divisions and apoptosis.
pub fn generateTrackingLineage(a: Allocator, n_total_cells: u32) !Tracking {
    var tracking = try std.ArrayList(TrackedCell).initCapacity(a, n_total_cells);
    var unfinished_lineage_q = try std.ArrayList(TrackedCell).initCapacity(a, 100);
    defer unfinished_lineage_q.deinit();

    // cumulative distribution of events
    const jumpdist: f32 = 0.015;
    const p_new_lineage: f32 = 0.00;
    // const p_continue: f32 = 0.97 + p_new_lineage;
    const p_continue: f32 = 1.0 + p_new_lineage;
    const p_divide: f32 = 0.020 + p_continue;
    const p_death: f32 = 0.010 + p_divide;
    _ = p_death;

    // Add ten cells to the starting frame
    for (0..10) |i| {
        const cell = TrackedCell{
            .pt = .{ @as(f32, @floatFromInt(i)) / 10.0, 0.1 },
            // .pt = .{ random.float(f32), random.float(f32) },
            .id = @as(u32, @intCast(tracking.items.len)),
            .time = 0, //random.uintLessThan(u16, 4),
            .parent_id = null,
        };
        try unfinished_lineage_q.append(cell);
        try tracking.append(cell);
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
                .id = @as(u32, @intCast(tracking.items.len)),
                .time = 0, //random.uintLessThan(u16, 4),
                .parent_id = null,
            };
            try unfinished_lineage_q.append(cell);
            try tracking.append(cell);
        } else if (rchoice < p_continue) {
            // 80% chance we continue the parent lineage

            const dx = 0.000 + 3 * jumpdist * (0.5 - random.float(f32));
            const dy = 0.005 + jumpdist * (0.5 - random.float(f32));

            // parent = if (unfinished_lineage_q.popOrNull()) |p| p else continue;

            const parent = unfinished_lineage_q.pop();
            const cell = .{
                // .pt = .{ parent.pt[0] + dx, parent.pt[1] + dy },
                .pt = .{ @mod(parent.pt[0] + dx, 1), @mod(parent.pt[1] + dy, 1) },
                .id = @as(u32, @intCast(tracking.items.len)),
                .time = parent.time + 1,
                .parent_id = parent.id,
            };
            try unfinished_lineage_q.append(cell);
            try tracking.append(cell);
        } else if (rchoice < p_divide) {
            // 5% chance we divide. pop once, enqueue twice.
            // Daughter cells appear on opposite sides of the mother, equally spaced

            const parent = unfinished_lineage_q.pop();
            const dx = jumpdist * (0.5 - random.float(f32));
            const dy = jumpdist * (0.5 - random.float(f32));

            // create and enqueue cell 1
            const cell1 = .{
                .pt = .{ @mod(parent.pt[0] + dx, 1), @mod(parent.pt[1] + dy, 1) },
                .id = @as(u32, @intCast(tracking.items.len)),
                .time = parent.time + 1,
                .parent_id = parent.id,
            };
            try unfinished_lineage_q.append(cell1);
            try tracking.append(cell1);

            if (tracking.items.len == tracking.capacity) break;

            // create and enqueue cell 2
            const cell2 = .{
                .pt = .{ @mod(parent.pt[0] - dx, 1), @mod(parent.pt[1] - dy, 1) },
                .id = @as(u32, @intCast(tracking.items.len)),
                .time = parent.time + 1,
                .parent_id = parent.id,
            };
            try unfinished_lineage_q.append(cell2);
            try tracking.append(cell2);
        } else {
            // 5% chance the lineage dies

            _ = unfinished_lineage_q.pop();
        }
    }

    // This fails to compile because calling free on this slice later
    return .{ .items = tracking.items };
}
