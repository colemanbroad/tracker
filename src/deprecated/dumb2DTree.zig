/// This tree doesn't hold any points or have any automated construction method based on points.
/// Based on `clbgTrees.zig`
/// 

const std = @import("std");
// var allo = std.testing.allocator;
// var allocator = std.heap.c_allocator;
var allocator = std.testing.allocator;
const print = std.debug.print;
const geo = @import("geometry.zig");

const Allocator = std.mem.Allocator;
const Vec2 = geo.Vec2;

test "alloc and free" {
    const tn = try allocator.alloc(TreeNode, 1);
    defer allocator.free(tn);

    const tn2 = try allocator.create(TreeNode);
    defer allocator.destroy(tn2);
}

const TreeNode = struct {
    tl: ?*TreeNode,
    tr: ?*TreeNode,
    bl: ?*TreeNode,
    br: ?*TreeNode,

    pub fn new(
        a: Allocator,
        tl: ?*TreeNode,
        tr: ?*TreeNode,
        bl: ?*TreeNode,
        br: ?*TreeNode,
    ) !*TreeNode {
        var node = try a.create(TreeNode);
        node.tl = tl;
        node.tr = tr;
        node.bl = bl;
        node.br = br;
        return node;
    }

    pub fn free(self: *TreeNode, a: Allocator) void {
        a.destroy(self);
    }
};

/// count nodes in tree. include root!
fn itemCheck(node: *TreeNode) usize {
    var res: usize = 0;

    if (node.tl) |x| res += 1 + itemCheck(x);
    if (node.tr) |x| res += 1 + itemCheck(x);
    if (node.bl) |x| res += 1 + itemCheck(x);
    if (node.br) |x| res += 1 + itemCheck(x);

    return res + 1;
}

fn bottomUpTree(a: Allocator, depth: usize) Allocator.Error!*TreeNode {
    if (depth > 0) {
        const tl = try bottomUpTree(a, depth - 1);
        const tr = try bottomUpTree(a, depth - 1);
        const bl = try bottomUpTree(a, depth - 1);
        const br = try bottomUpTree(a, depth - 1);

        return try TreeNode.new(
            a,
            tl,
            tr,
            bl,
            br,
        );
    } else {
        return try TreeNode.new(
            a,
            null,
            null,
            null,
            null,
        );
    }
}

fn deleteTree(a: Allocator, node: *TreeNode) void {
    if (node.tl) |x| deleteTree(a, x);
    if (node.tr) |x| deleteTree(a, x);
    if (node.bl) |x| deleteTree(a, x);
    if (node.br) |x| deleteTree(a, x);
    a.destroy(node);
}

// pub fn main() !void {
test "trees. build tree and count nodes" {

    // var args = std.process.args();
    // _ = args.skip();
    // const n = try std.fmt.parseUnsigned(u8, args.next().?, 10);
    const n = 4;

    print("sum = {}\n", .{4 * 4 * 4 * 4 * 4 + 4 * 4 * 4 * 4 + 4 * 4 * 4 + 4 * 4 + 4 + 1});

    const min_depth: usize = 4;
    const max_depth: usize = n;
    const stretch_depth = max_depth + 1;

    const stretch_tree = try bottomUpTree(allocator, stretch_depth);
    print("depth {}, check {}\n", .{ stretch_depth, itemCheck(stretch_tree) });
    deleteTree(allocator, stretch_tree);

    const long_lived_tree = try bottomUpTree(allocator, max_depth);
    var depth = min_depth;
    while (depth <= max_depth) : (depth += 2) {
        var iterations = @floatToInt(usize, std.math.pow(f32, 2, @intToFloat(f32, max_depth - depth + min_depth)));
        var check: usize = 0;

        var i: usize = 1;
        while (i <= iterations) : (i += 1) {
            const temp_tree = try bottomUpTree(allocator, depth);
            check += itemCheck(temp_tree);
            deleteTree(allocator, temp_tree);
        }

        print("{} trees of depth {}, check {}\n", .{ iterations, depth, check });
    }

    print("long lived tree of depth {}, check {}\n", .{ max_depth, itemCheck(long_lived_tree) });
    deleteTree(allocator, long_lived_tree);
}
