const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;
const im = @import("image_base.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var allocator = gpa.allocator();
// var allocator = std.testing.allocator;
var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

const Tracer = @import("fn-tracer.zig").Tracer;
var tracer: Tracer(Ntrials) = undefined;

// Get an SDL window we can use for visualizing algorithms.
const sdlw = @import("sdl-window.zig");
var win: sdlw.Window = undefined;
var running_as_main_use_sdl = false;

const V2 = @Vector(2, f32);
fn v2(x: [2]f32) V2 {
    return x;
}

const N = 10_001;
const Ntrials = 5_000;

const Pt = [2]f32;
const Volume = struct { min: Pt, max: Pt };

// Consider this as an alternative to the KDNode that avoids the issues with
// null. We need to reference points by an index and array, so that we can
// distinguish between two points with identical coordinates.
const NodeTag = enum { Split, Leaf, Empty };
const NodeUnion = union(NodeTag) {
    Split: struct { pt: Pt, l: *NodeUnion, r: *NodeUnion, dim: u8, vol: Volume },
    Leaf: struct { pt: Pt, vol: Volume },
    Empty: struct { vol: Volume },

    pub fn format(self: NodeUnion, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        // try writer.print("{s:9} {d:>10.0}\n", .{ "mean", self.mean });
        // try writer.print("{s:9} {d:>10.0}\n", .{ "median", self.median });
        // try writer.print("{s:9} {d:>10.0}\n", .{ "mode", self.mode });
        // try writer.print("{s:9} {d:>10.0}\n", .{ "min", self.min });
        // try writer.print("{s:9} {d:>10.0}\n", .{ "max", self.max });
        // try writer.print("{s:9} {d:>10.0}\n", .{ "std dev", self.stddev });

        const at = std.meta.activeTag;

        switch (self) {
            .Empty => |_| try writer.print("Empty \n", .{}),
            .Leaf => |l| try writer.print("Leaf {d} \n", .{l.pt}),
            .Split => |s| {
                try writer.print("Split {d:0.3} {s} {s} \n", .{ s.pt, @tagName(at(s.l.*)), @tagName(at(s.r.*)) });
            },
        }

        try writer.writeAll("");
    }
};

const NodeIdx = union(NodeTag) {
    Split: struct { pt: Pt, l: *NodeIdx, r: *NodeIdx, dim: u8, vol: Volume, idx: usize },
    Leaf: struct { pt: Pt, vol: Volume, idx: usize },
    Empty: struct { vol: Volume },
};

fn ltPtsIdx(idx: u8, lhs: Pt, rhs: Pt) bool {
    return lhs[idx] < rhs[idx];
}

pub fn dist(a: Pt, b: Pt) f32 {
    // return std.math.absFloat(a - b);
    const d = Pt{ a[0] - b[0], a[1] - b[1] };
    const norml2 = @sqrt(d[0] * d[0] + d[1] * d[1]);
    return norml2;
}

pub fn distSquared(a: Pt, b: Pt) f32 {
    // return std.math.absFloat(a - b);
    const d = Pt{ a[0] - b[0], a[1] - b[1] };
    const norml2 = d[0] * d[0] + d[1] * d[1];
    return norml2;
}

fn ndMinMax(pts: []Pt) Volume {
    // if (pts.len == 0) return error.InvalidArraySize;
    // if (pts.len <= 1) return error.InvalidArraySize;
    // !!!WARN : Skip this check. It's always performed in buildTree.

    var p_min = pts[0];
    var p_max = pts[0];
    for (pts) |p| {
        for (p, 0..) |x_i, i| {
            p_min[i] = @min(x_i, p_min[i]);
            p_max[i] = @max(x_i, p_max[i]);
        }
    }
    return Volume{ .min = p_min, .max = p_max };
}

// var globocount: u32 = 0;

// This alternative implementation sorts indices instead of points directly, allowing us
// to record index locations (needed for tracking). Actually, we can skip this work, because
// we already know that the sorted list of points is a better NN datastructure for us!
fn buildTreeIndex(pts: []Pt, bounding_volume: Volume) !*NodeUnion {
    _ = bounding_volume;
    _ = pts;
}

fn buildTree(pts: []Pt, bounding_volume: Volume) !*NodeUnion {
    const a = allocator;

    // globocount += 1;
    // print("globocount = {d} \n", .{globocount});

    // We have to copy the points to avoid problems when sorting and assigning.
    // NO CHANGE
    // var pts = try a.alloc(Pt, _pts.len);
    // defer a.free(pts);
    // for (pts, 0..) |*p, i| p.* = _pts[i];

    var node = try a.create(NodeUnion);

    // if pts.len == 0: return Empty and
    // if pts.len == 1: return Leaf and bounding_volume.

    if (pts.len == 0) {
        node.* = .{ .Empty = .{ .vol = bounding_volume } };
        return node;
    }
    if (pts.len == 1) {
        node.* = .{ .Leaf = .{ .pt = pts[0], .vol = bounding_volume } };
        return node;
    }

    // else: compute bounding box on pts
    const span = ndMinMax(pts);

    // find median point P along the axis with greater variance
    const idx = ptArgmax(v2(span.max) - v2(span.min));
    std.sort.heap(Pt, pts, idx, ltPtsIdx);
    const midpoint = pts[pts.len / 2];

    // Update bounding volumes for both sides of the split

    // TODO: I don't think @min/@max is necessary ?
    var l_vol = bounding_volume;
    l_vol.max[idx] = midpoint[idx];
    var r_vol = bounding_volume;
    r_vol.min[idx] = midpoint[idx];

    // create a new node with P as split

    node.* = .{ .Split = .{
        .pt = midpoint,
        .l = try buildTree(pts[0 .. pts.len / 2], l_vol),
        .r = try buildTree(pts[pts.len / 2 + 1 ..], r_vol),
        .dim = idx,
        .vol = bounding_volume,
    } };

    return node;
}

const GridHashFlex = struct {
    const NXbins = 10; // We split the X dimension up into 10th-percentiles.
    const NYbins = 10; // Do the same for Y, for now.
    dim1_bucket_bounds: [NXbins + 1]f32,
    dim2_bucket_bounds: [NXbins][NYbins + 1]f32, // Each X bin has unique Y bin bucket bounds
    data: []Pt, // Assume Data is sorted
    fn bucketToSlice() []Pt {}
};

fn argmax(comptime T: type, arr: []T) struct { max: T, arg: u32 } {
    var max = arr[0];
    var arg = 0;
    for (arr, 0..) |x, i| {
        if (x > max) {
            max = x;
            arg = i;
        }
    }
    return .{ .max = max, .arg = arg };
}

// !!! WARN TODO only works with ndim==2
fn ptArgmax(v: Pt) u8 {
    const m = @max(v[0], v[1]);
    if (v[0] == m) return 0;
    return 1;
}

fn randomColor() [4]u8 {
    return .{
        random.int(u8),
        random.int(u8),
        255,
        255,
    };
}

fn drawTree(root: *NodeUnion) !void {
    const ParentNode = struct { parent: ?[2]i32, node: *NodeUnion };
    var node_q: [2 * N]ParentNode = undefined;

    var h_idx: usize = 0; // head index
    var t_idx: usize = 0; // tail index

    // const TupT = struct { i32, i32, [2][2]i32 };
    // var point_list: [100]TupT = undefined;
    // var point_list_idx: usize = 0;

    const Counts = struct { empty: u16 = 0, leaf: u16 = 0, split: u16 = 0 };
    var counts = Counts{};

    node_q[h_idx] = .{ .parent = null, .node = root };
    h_idx += 1;

    while (t_idx < h_idx) {
        win.awaitKeyPressAndUpdateWindow();

        const p_and_n = node_q[t_idx];
        t_idx += 1;
        const n = p_and_n.node.*;
        const parent_pt = p_and_n.parent;
        _ = parent_pt;

        if (n == .Empty) {
            counts.empty += 1;
            continue;
        }

        const n_pt = switch (n) {
            .Leaf => |l| l.pt,
            .Split => |l| l.pt,
            else => unreachable,
        };

        const x = @as(i32, @intFromFloat(n_pt[0] * 750 + 25));
        const y = @as(i32, @intFromFloat(n_pt[1] * 750 + 25));

        print("{any}", .{n});

        im.drawCircle([4]u8, win.pix, x, y, 3, .{ 255, 255, 255, 255 });

        // Draw the Vertical / Horizontal split line
        if (n == .Split) {
            const bv = n.Split.vol;

            const line: [2][2]i32 = switch (n.Split.dim) {
                0 => .{
                    .{ x, @as(i32, @intFromFloat(bv.min[1] * 750 + 25)) },
                    .{ x, @as(i32, @intFromFloat(bv.max[1] * 750 + 25)) },
                },
                1 => .{
                    .{ @as(i32, @intFromFloat(bv.min[0] * 750 + 25)), y },
                    .{ @as(i32, @intFromFloat(bv.max[0] * 750 + 25)), y },
                },
                else => unreachable,
            };

            im.drawLineInBounds([4]u8, win.pix, line[0][0], line[0][1], line[1][0], line[1][1], randomColor());
            // point_list[point_list_idx] = .{ x, y, line };
            // point_list_idx += 1;

        }

        if (n == .Leaf) {
            counts.leaf += 1;
            continue;
        }

        counts.split += 1;

        // TERRIBLE BUG!!! DONT TAKE POINTERS TO LOCAL LOOP VARIABLES!!!
        // var n_l = n.Split.l.*;
        // _ = n_l;
        // var n_r = n.Split.r.*;
        // _ = n_r;

        node_q[h_idx] = .{ .parent = .{ x, y }, .node = n.Split.l };
        h_idx += 1;
        node_q[h_idx] = .{ .parent = .{ x, y }, .node = n.Split.r };
        h_idx += 1;
    }
}

const NNReturn = struct { dist: f32, pt: Pt, node: NodeUnion };

// TODO: be more efficient than 2*N here. shouldn't be nearly that many.
pub fn findNearestNeibKDTree(tree_root: *NodeUnion, query_point: Pt) NNReturn {
    const tspan = tracer.start(@src().fn_name);
    defer tspan.stop();

    var current_min_dist = dist(tree_root.Split.pt, query_point);
    var current_min_node = tree_root.*;
    var current_pt = tree_root.Split.pt;

    var itercount: u32 = 0;

    // First walk down the tree to find a point somewhere close to the query.
    // This cuts the avg number of traversals down significantly!
    var traversing_node = tree_root.*;
    while (traversing_node == .Split) {
        itercount += 1;

        const n = traversing_node.Split;
        const next_dist = dist(n.pt, query_point);
        if (next_dist < current_min_dist) {
            current_min_dist = next_dist;
            current_min_node = traversing_node;
            current_pt = n.pt;
        }

        // Walk down the tree in the direction of query_point
        if (n.pt[n.dim] < query_point[n.dim]) {
            traversing_node = n.r.*;
        } else {
            traversing_node = n.l.*;
        }
    }

    // Test it again just in case the last node was a leaf!
    if (traversing_node == .Leaf) {
        const n = traversing_node.Leaf;
        const next_dist = dist(n.pt, query_point);
        if (next_dist < current_min_dist) {
            current_min_dist = next_dist;
            current_min_node = traversing_node;
            current_pt = n.pt;
        }
    }

    // Now we do the full-tree depth-first search with pruing using an explicit stack.
    // 100 is the maximum possible stack depth. Empirically the max h_idx for 1_000_000 pts is 5.
    var node_stack: [100]NodeUnion = undefined;
    var stack_idx: u32 = 0;

    node_stack[stack_idx] = tree_root.*;
    stack_idx += 1;

    // var h_idx_max: u32 = 0;
    // _ = h_idx_max;

    while (stack_idx > 0) {

        // if (h_idx > h_idx_max) {
        //     h_idx_max = h_idx;
        //     print("h_idx = {d}\n", .{h_idx});
        // }

        itercount += 1;
        // WARN: remember that h_idx points to the next empty spot (insert position)
        // but the last full position is at the previous index!
        const next_node = node_stack[stack_idx - 1];
        stack_idx -= 1;

        // coalesce .Leaf and .Split into one
        const pt = switch (next_node) {
            .Leaf => |l| l.pt,
            .Split => |l| l.pt,
            else => continue,
        };
        const next_dist = dist(pt, query_point);
        if (next_dist < current_min_dist) {
            current_min_dist = next_dist;
            current_min_node = next_node;
            current_pt = pt;
        }

        if (next_node != .Split) continue;

        const s = next_node.Split;
        const orthogonal_distance = s.pt[s.dim] - query_point[s.dim];

        // If the orthogonal displacement is greater than the current best distance (D)
        // then we know we can avoid any points on the greater side. If the orthogonal
        // displacement is less than -D, then we can avoid any points on the lesser side.

        // Walk down the tree in the direction of query_point
        // If you're far away from the query point you can completly avoid some branches
        // If you're within the query point current best radius then you first search
        // down the closer side by adding it 2nd in the stack.
        if (orthogonal_distance < -current_min_dist) {
            node_stack[stack_idx] = s.r.*;
            stack_idx += 1;
        } else if (orthogonal_distance < 0) {
            node_stack[stack_idx] = s.l.*;
            stack_idx += 1;
            node_stack[stack_idx] = s.r.*;
            stack_idx += 1;
        } else if (orthogonal_distance < current_min_dist) {
            node_stack[stack_idx] = s.r.*;
            stack_idx += 1;
            node_stack[stack_idx] = s.l.*;
            stack_idx += 1;
        } else {
            node_stack[stack_idx] = s.l.*;
            stack_idx += 1;
        }

        // if (!(orthogonal_distance > current_min_dist)) {
        //     // add s.r
        //     node_stack[stack_idx] = s.r.*;
        //     stack_idx += 1;
        // }
        // if (!(orthogonal_distance < -current_min_dist)) {
        //     // add s.l
        //     node_stack[stack_idx] = s.l.*;
        //     stack_idx += 1;
        // }
    }

    // print("itercount = {d}\n", .{itercount});
    return .{ .dist = current_min_dist, .pt = current_pt, .node = current_min_node };
}

pub fn greatestLowerBoundIndex(pts: []Pt, dim: u8, query_point: Pt) !usize {
    const tspan = tracer.start(@src().fn_name);
    defer tspan.stop();

    const target_val = query_point[dim];

    var lower_bound_idx: usize = 0; // inclusive
    var upper_bound_idx: usize = pts.len; // exclusive

    // INVALID INDEX ENCODES
    if (query_point[dim] < pts[0][dim]) return error.IndexOOB;
    if (query_point[dim] > pts[pts.len - 1][dim]) return pts.len - 1;

    while (true) {

        // First look in the middle. Rounds down so (1 + 0) / 2 == 0, but
        // (2 + 0) / 2 == 1, so the median falls to the right side.
        var current_idx = (upper_bound_idx + lower_bound_idx) / 2;
        var current_val = pts[current_idx][dim];

        // print("{any} \n", .{.{ .low = lower_bound_idx, .hi = upper_bound_idx, .cur = current_idx }});

        // If current_val < target_val then we need to move the lower bound up.
        if (current_val < target_val) {
            lower_bound_idx = current_idx + 1; // inclusive
        } else if (current_val == target_val) {
            return current_idx;
        } else {
            upper_bound_idx = current_idx; // don't try this index
        }

        if (lower_bound_idx == upper_bound_idx) return lower_bound_idx;
    }
}

test "test greatestLowerBoundIndex" {
    print("\n", .{});
    const a = allocator;
    var pts = try a.alloc(Pt, N);
    defer a.free(pts);
    for (pts) |*v| v.* = .{ random.float(f32), random.float(f32) };

    const dim_idx: u8 = 0;
    std.sort.heap(Pt, pts, dim_idx, ltPtsIdx);
    // const query_point = .{ random.float(f32), random.float(f32) };

    {
        const query_point = pts[999];
        const glb_idx = try greatestLowerBoundIndex(pts, dim_idx, query_point);
        try std.testing.expect(glb_idx == 999);
    }

    {
        const query_point = pts[998];
        const glb_idx = try greatestLowerBoundIndex(pts, dim_idx, query_point);
        try std.testing.expect(glb_idx == 998);
    }

    {
        const query_point = pts[997];
        const glb_idx = try greatestLowerBoundIndex(pts, dim_idx, query_point);
        try std.testing.expect(glb_idx == 997);
    }
    {
        const query_point = pts[0];
        const glb_idx = try greatestLowerBoundIndex(pts, dim_idx, query_point);
        try std.testing.expect(glb_idx == 0);
    }
    {
        const query_point = pts[N - 1];
        const glb_idx = try greatestLowerBoundIndex(pts, dim_idx, query_point);
        try std.testing.expect(glb_idx == N - 1);
    }

    {
        var query_point = pts[0];
        query_point[0] -= 0.1;
        const glb_idx = greatestLowerBoundIndex(pts, dim_idx, query_point);
        try std.testing.expect(glb_idx == error.IndexOOB);
    }

    {
        var query_point = pts[pts.len - 1];
        query_point[0] += 0.1;
        const glb_idx = try greatestLowerBoundIndex(pts, dim_idx, query_point);
        try std.testing.expect(glb_idx == pts.len - 1);
    }
}

const assert = std.debug.assert;

pub fn greatestLowerBoundIndexGeneric(comptime T: type, pts: []const T, dim: u8, query_point: Pt) !usize {
    const tspan = tracer.start(@src().fn_name);
    defer tspan.stop();

    const target_val = query_point[dim];

    var lower_bound_idx: usize = 0; // inclusive
    var upper_bound_idx: usize = pts.len; // exclusive

    // INVALID INDEX ENCODES
    if (query_point[dim] < pts[0].pt[dim]) return error.IndexOOB;
    if (query_point[dim] > pts[pts.len - 1].pt[dim]) return pts.len - 1;

    while (true) {

        // First look in the middle. Rounds down so (1 + 0) / 2 == 0, but
        // (2 + 0) / 2 == 1, so the median falls to the right side.
        var current_idx = (upper_bound_idx + lower_bound_idx) / 2;
        var current_val = pts[current_idx].pt[dim];

        // print("{any} \n", .{.{ .low = lower_bound_idx, .hi = upper_bound_idx, .cur = current_idx }});

        // If current_val < target_val then we need to move the lower bound up.
        if (current_val < target_val) {
            lower_bound_idx = current_idx + 1; // inclusive
        } else if (current_val == target_val) {
            return current_idx;
        } else {
            upper_bound_idx = current_idx; // don't try this index
        }

        if (lower_bound_idx == upper_bound_idx) return lower_bound_idx;
    }
}

pub fn findNearestNeibFromSortedListGeneric(comptime T: type, pts: []const T, query_point: Pt) usize {
    const tspan = tracer.start(@src().fn_name);
    defer tspan.stop();

    assert(@hasField(T, "pt"));

    // First we find the greatest lower bound (index)
    const dim_idx = 0;
    const glb_idx = greatestLowerBoundIndexGeneric(T, pts, dim_idx, query_point) catch 0;

    var color: [4]u8 = undefined;

    // Now we know where to start our search. Get the d_euclid to query_point,
    // and search outwards along the sorted dimension. You can stop searching
    // when the d_axis of current point > d_euclid_best.

    var current_best_dist = dist(pts[glb_idx].pt, query_point);
    var current_best_idx = glb_idx;
    var idx_offset: u16 = 0;

    // First search to the left.
    while (true) {
        const idx = glb_idx - idx_offset;
        const current_pt = pts[idx].pt;
        if (query_point[0] == 0.0855857804 and current_pt[0] == 0.0828876867) @breakpoint();
        const d_axis = current_pt[dim_idx] - query_point[dim_idx];
        if (d_axis < -current_best_dist) break;
        const d_euclid = dist(current_pt, query_point);
        // WARNING: TODO: make all nn methods consistent when multiple points
        // have the same distance.

        color = .{ 255, 255, 0, 255 };

        if (d_euclid < current_best_dist) {
            current_best_dist = d_euclid;
            current_best_idx = idx;
            color = .{ 0, 255, 255, 255 };
        }

        if (running_as_main_use_sdl) {
            im.drawLineInBounds(
                [4]u8,
                win.pix,
                @as(i32, @intFromFloat(query_point[0] * 750 + 25)),
                @as(i32, @intFromFloat(query_point[1] * 750 + 25)),
                @as(i32, @intFromFloat(current_pt[0] * 750 + 25)),
                @as(i32, @intFromFloat(current_pt[1] * 750 + 25)),
                color,
            );
            win.awaitKeyPressAndUpdateWindow();
        }

        if (idx == 0) break;
        idx_offset += 1;
    }

    idx_offset = 0;

    // Then search to the right.

    while (true) {
        const idx = glb_idx + idx_offset;
        if (idx == pts.len) break;
        const current_pt = pts[idx].pt;
        const d_axis = current_pt[dim_idx] - query_point[dim_idx];

        if (d_axis > current_best_dist) {
            break;
        }

        color = .{ 255, 255, 0, 255 };
        const d_euclid = dist(current_pt, query_point);
        if (d_euclid < current_best_dist) {
            current_best_dist = d_euclid;
            current_best_idx = idx;
            color = .{ 0, 255, 255, 255 };
        }

        if (running_as_main_use_sdl) {
            im.drawLineInBounds(
                [4]u8,
                win.pix,
                @as(i32, @intFromFloat(query_point[0] * 750 + 25)),
                @as(i32, @intFromFloat(query_point[1] * 750 + 25)),
                @as(i32, @intFromFloat(current_pt[0] * 750 + 25)),
                @as(i32, @intFromFloat(current_pt[1] * 750 + 25)),
                color,
            );

            win.awaitKeyPressAndUpdateWindow();
        }

        idx_offset += 1;
    }

    return current_best_idx;
    // return pts[current_best_idx];
    // return .{ .pt = pts[current_best_idx], .dist = current_best_dist };
}

pub fn findNearestNeibFromSortedList(pts: []Pt, query_point: Pt) usize {
    const tspan = tracer.start(@src().fn_name);
    defer tspan.stop();

    // First we find the greatest lower bound (index)
    const dim_idx = 0;
    const glb_idx = greatestLowerBoundIndex(pts, dim_idx, query_point) catch 0;

    var color: [4]u8 = undefined;

    // Now we know where to start our search. Get the d_euclid to query_point,
    // and search outwards along the sorted dimension. You can stop searching
    // when the d_axis of current point > d_euclid_best.

    var current_best_dist = dist(pts[glb_idx], query_point);
    var current_best_idx = glb_idx;
    var idx_offset: u16 = 0;

    // First search to the left.
    while (true) {
        const idx = glb_idx - idx_offset;
        const current_pt = pts[idx];
        if (query_point[0] == 0.0855857804 and current_pt[0] == 0.0828876867) @breakpoint();
        const d_axis = current_pt[dim_idx] - query_point[dim_idx];
        if (d_axis < -current_best_dist) break;
        const d_euclid = dist(current_pt, query_point);
        // WARNING: TODO: make all nn methods consistent when multiple points
        // have the same distance.

        color = .{ 255, 255, 0, 255 };

        if (d_euclid < current_best_dist) {
            current_best_dist = d_euclid;
            current_best_idx = idx;
            color = .{ 0, 255, 255, 255 };
        }

        if (running_as_main_use_sdl) {
            im.drawLineInBounds(
                [4]u8,
                win.pix,
                @as(i32, @intFromFloat(query_point[0] * 750 + 25)),
                @as(i32, @intFromFloat(query_point[1] * 750 + 25)),
                @as(i32, @intFromFloat(current_pt[0] * 750 + 25)),
                @as(i32, @intFromFloat(current_pt[1] * 750 + 25)),
                color,
            );
            win.awaitKeyPressAndUpdateWindow();
        }

        if (idx == 0) break;
        idx_offset += 1;
    }

    idx_offset = 0;

    // Then search to the right.

    while (true) {
        const idx = glb_idx + idx_offset;
        if (idx == pts.len) break;
        const current_pt = pts[idx];
        const d_axis = current_pt[dim_idx] - query_point[dim_idx];

        if (d_axis > current_best_dist) {
            break;
        }

        color = .{ 255, 255, 0, 255 };
        const d_euclid = dist(current_pt, query_point);
        if (d_euclid < current_best_dist) {
            current_best_dist = d_euclid;
            current_best_idx = idx;
            color = .{ 0, 255, 255, 255 };
        }

        if (running_as_main_use_sdl) {
            im.drawLineInBounds(
                [4]u8,
                win.pix,
                @as(i32, @intFromFloat(query_point[0] * 750 + 25)),
                @as(i32, @intFromFloat(query_point[1] * 750 + 25)),
                @as(i32, @intFromFloat(current_pt[0] * 750 + 25)),
                @as(i32, @intFromFloat(current_pt[1] * 750 + 25)),
                color,
            );

            win.awaitKeyPressAndUpdateWindow();
        }

        idx_offset += 1;
    }

    return current_best_idx;
    // return pts[current_best_idx];
    // return .{ .pt = pts[current_best_idx], .dist = current_best_dist };
}

test "simple union experiment" {
    var a: NodeUnion = undefined;

    if (true) return;

    if (random.float(f32) < 0.5) {
        a = NodeUnion{ .Empty = .{ .vol = .{ .min = .{ 0, 0 }, .max = .{ 2, 2 } } } };
    } else {
        a = NodeUnion{ .Leaf = .{ .pt = .{ 8, 8 }, .vol = .{ .min = .{ 0, 0 }, .max = .{ 2, 2 } } } };
    }

    // ERROR! "Incompatible types"
    const b = switch (a) {
        .Empty => |x| x,
        .Leaf => |x| x,
        .Split => |x| x.pt,
    };

    print("{any} \n", .{b});
}

test "test build a Tree" {
    print("\n", .{});
    var a = allocator;

    var pts = try a.alloc(Pt, 13);
    defer a.free(pts);
    for (pts) |*v| v.* = .{ random.float(f32), random.float(f32) };

    var q = try buildTree(pts[0..], .{ .min = .{ 0, 0 }, .max = .{ 1, 1 } });

    // @breakpoint();
    std.debug.print("{any}", .{q.*});

    // defer deleteNodeAndChildren(a, q);
    // try printTree(a, q, 0);
}

pub fn findNearestNeibBruteForce(pts: []Pt, query_point: Pt) usize {
    const tspan = tracer.start(@src().fn_name);
    defer tspan.stop();

    var closest_pt: Pt = pts[0];
    var closest_dist: f32 = dist(query_point, pts[0]);
    var closest_idx: usize = 0;

    for (pts, 0..) |p, idx| {
        // me too
        const d = dist(query_point, p);

        // test me out
        if (d < closest_dist) {
            closest_pt = p;
            closest_dist = d;
            closest_idx = idx;
        }
    }
    return closest_idx;
}

pub fn main() !u8 {
    running_as_main_use_sdl = true;

    // traces = std.ComptimeStringMap(comptime V: type, comptime kvs_list: anytype)
    // traces = std.StringHashMap(Trace).init(allocator);
    tracer = try Tracer(Ntrials).init();

    // Create random points
    const a = allocator;
    var pts = try a.alloc(Pt, N);
    defer a.free(pts);
    for (pts) |*v| v.* = .{ random.float(f32), random.float(f32) };

    // Initialize Drawing Pallete
    if (running_as_main_use_sdl) {
        try sdlw.initSDL();
        win = try sdlw.Window.init(1000, 800);
        // win.markBounds();
        for (pts) |p| {
            const x = @as(i32, @intFromFloat(p[0] * 750 + 25));
            const y = @as(i32, @intFromFloat(p[1] * 750 + 25));
            im.drawCircle([4]u8, win.pix, x, y, 3, .{ 255, 255, 255, 255 });
        }
        try win.update();
    }
    defer if (running_as_main_use_sdl) sdlw.quitSDL();

    // Build the KDTree
    var tree_root = try buildTree(pts[0..], .{ .min = .{ 0, 0 }, .max = .{ 1, 1 } });

    // try drawTree(tree_root);

    // Sort the points along the X dimension.
    const dim_idx: u8 = 0;
    std.sort.heap(Pt, pts, dim_idx, ltPtsIdx);

    for (0..Ntrials) |_| {
        // while (true) {
        const query_point = Pt{ random.float(f32), random.float(f32) };

        const nn_kdtree = findNearestNeibKDTree(tree_root, query_point).pt;
        const nn_brute_force_idx = findNearestNeibBruteForce(pts, query_point);
        const nn_brute_force = pts[nn_brute_force_idx];

        // if (query_point[0] == 0.0855857804) @breakpoint();
        const nn_bsorted_idx = findNearestNeibFromSortedList(pts, query_point);
        const nn_bsorted = pts[nn_bsorted_idx];

        std.testing.expectEqualDeep(nn_kdtree, nn_bsorted) catch @breakpoint();
        std.testing.expectEqualDeep(nn_kdtree, nn_brute_force) catch @breakpoint();

        //     print("ERROR #1 PTS MISSING\n{d}\n{d}\n{d}\n{d}\n", .{ query_point, nn_kdtree, nn_brute_force, nn_bsorted });
        // };
        // std.testing.expectEqualDeep(nn_kdtree, nn_brute_force) catch {};
        // const x1 = nn_bsorted[0];
        // std.debug.print("x1 = {d}\n", .{x1});
        //     print("ERROR #2 PTS MISSING\n{d}\n{d}\n{d}\n{d}\n", .{ query_point, nn_kdtree, nn_brute_force, nn_bsorted });
        // };
    }

    try tracer.analyze(allocator);

    return 0;
}
