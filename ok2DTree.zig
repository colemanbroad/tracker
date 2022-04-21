// still need to do: be able to add points and detect intersection with BBoxes.
// to construct a tree i pass in a list of BBoxes and get back a tree.
// i can query the tree with a new point / bbox and determine containment / overlap.
// i can add new bboxes to the tree.

// Q: is it _enough_ to determine bbox containment for the purpose of delaunay construction?
//    The bounding box of a triangle is is not contained within the bounding circle (or vice versa).
//    To find a tri that contains pt p, we can still construct a tree that branches tris by centroid location,
//    then once we've found tri with closes centroid location to `p` we can search backwards through tree.
//    We only need to find one hit for Delaunay. Then we can do the remaining search in the triangle graph (voronoi graph).

// return the root node
// split triangles into equally balanced piles.
// splits alternate between vertical and horizontal ? or do we always split both ways?
// compute point for each tri, e.g. midpoint.

// A tree that automatically splits 2D points up into vertical and horizontal segments.
//
//
// TODO:
// - split pts into LEFT PTS < PiVoT <= RIGHT PTS with a single pass. use std.ArrayList
// - how to determine pivot? find median by sorting points. then split.
//
//

const std = @import("std");
const geo = @import("geometry.zig");

const print = std.debug.print;

const Allocator = std.mem.Allocator;
const Vec2 = geo.Vec2;
const Tri = @import("delaunay.zig").Tri;

var allocator = std.testing.allocator;
var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

const XY = enum {
    X,
    Y,
};

const Pt = [2]f32;

// IDEA merges vals and splt into union ?
const KDNode = struct {
    l: ?*KDNode, // less than pivot
    r: ?*KDNode, // greater than pivot
    dim: XY, // just ignore it if we're at a leaf
    vals: ?[]Pt,
    splt: ?Pt,
};

fn ltPtsDims(
    dim: XY,
    lhs: Pt,
    rhs: Pt,
) bool {
    switch (dim) {
        .X => return lhs[0] < rhs[0],
        .Y => return lhs[1] < rhs[1],
    }
}

fn buildTree(a: Allocator, pts: []Pt, dim: XY) Allocator.Error!*KDNode {
    if (pts.len < 3) {
        var node = try a.create(KDNode);
        node.l = null;
        node.r = null;
        node.vals = pts;
        node.splt = null;
        node.dim = dim; // doesn't matter
        return node;
    }

    // sort by dimension
    std.sort.sort(Pt, pts, dim, ltPtsDims);

    const idx = pts.len / 2;
    const median = pts[pts.len / 2];

    var node = try a.create(KDNode);
    const next_split = switch (dim) {.X => XY.Y , .Y => XY.X};
    node.l = try buildTree(a, pts[0..idx], next_split);
    node.r = try buildTree(a, pts[idx + 1 ..],   next_split);
    node.splt = median;
    node.vals = null;
    node.dim = dim;
    return node;
}

fn deleteTree(a: Allocator, node: *KDNode) void {
    if (node.l) |x| deleteTree(a, x);
    if (node.r) |x| deleteTree(a, x);
    a.destroy(node);
}

fn dist(a: Pt, b: Pt) f32 {
    return std.math.absFloat(a - b);
}

fn findNearest(root: *KDNode, pt: Pt) Allocator.Error!Pt {
    var nearest_pt = root.splt.?;
    var best_dist = dist(pt, root.splt.?);
    var current = root;

    // descend down branches until we get to a leaf. keep track of nearest point at all times.
    // In 2D we 
    while (true) {

        // we've made it to a leaf node. almost done.
        if (current.vals) |vals| {
            for (vals) |next_pt| {
                const d = dist(pt, next_pt);
                if (d < best_dist) {
                    nearest_pt = next_pt;
                    best_dist = d;
                }
            }
            return nearest_pt;
        }

        const splt_pt = current.splt.?;

        // test against split value
        const d = dist(pt, splt_pt);
        if (d < best_dist) {
            nearest_pt = splt_pt;
            best_dist = d;
        }

        if (pt > splt_pt) {
            current = current.r.?;
        } else {
            current = current.l.?;
        }
    }
}

fn printTree(a: Allocator, root: *KDNode, lvl: u8) Allocator.Error!void {
    var lvlstr = try a.alloc(u8, lvl * 2);
    defer a.free(lvlstr);
    for (lvlstr) |*v| v.* = '-';
    print("{s}", .{lvlstr});

    if (root.vals) |vals| {
        print(" {d:.3} \n", .{vals});
        return;
    }

    // we know splt is true
    const splt = root.splt.?;

    print(" {} - {d:.3} \n", .{ root.dim, splt });

    try printTree(a, root.l.?, lvl + 1);
    try printTree(a, root.r.?, lvl + 1);
}

test "kdnode test" {
    print("\n", .{});
    var a = allocator;

    var pts = try a.alloc(Pt, 13);
    defer a.free(pts);
    for (pts) |*v| v.* = .{random.float(f32), random.float(f32)};

    // std.sort.sort(f32, pts, {}, comptime std.sort.asc(f32));
    var q = try buildTree(a, pts[0..], .X);
    defer deleteTree(a, q);

    try printTree(a, q, 0);
}

fn testNearest(a: Allocator, N: u32) !void {
    var pts = try a.alloc(f32, N);
    defer a.free(pts);
    for (pts) |*v| v.* = random.float(f32);

    std.sort.sort(f32, pts, {}, comptime std.sort.asc(f32));
    var q = try buildTree(a, pts[0..], .X);
    defer deleteTree(a, q);

    const nearest1 = try findNearest(q, pts[N / 2]);
    print("pt = {} , nearest1 = {} \n", .{ pts[N / 2], nearest1 });
    try std.testing.expectEqual(pts[N / 2], nearest1);

    const pt = random.float(f32);
    const nearest2 = try findNearest(q, pt);
    print("pt = {} , nearest2 = {} \n", .{ pt, nearest2 });
    try std.testing.expect(std.math.absFloat(pt - nearest2) < 10.0 / @intToFloat(f32, N));
}

test "find nearest test" {
    if (true) return error.SkipZigTest;
    print("\n", .{});
    var a = allocator;
    try testNearest(a, 3); // must be at least 3 or findNearest() fails
    try testNearest(a, 10);
    try testNearest(a, 100);
    try testNearest(a, 1000);
    try testNearest(a, 10000);
}
