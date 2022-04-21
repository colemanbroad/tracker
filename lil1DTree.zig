
const std = @import("std");
const geo = @import("geometry.zig");

const print = std.debug.print;

const Allocator = std.mem.Allocator;
const Vec2 = geo.Vec2;
const Tri = @import("delaunay.zig").Tri;

var allocator = std.testing.allocator;

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();


const KDNode = struct {
    l: ?*KDNode,
    r: ?*KDNode,
    vals: ?[]f32,
    splt: ?f32,
};

fn buildTree(a:Allocator,pts:[]f32,) Allocator.Error!*KDNode {

    if (pts.len<3){
        var node = try a.create(KDNode);
        node.l = null;
        node.r = null;
        node.vals = pts;
        node.splt = null;
        return node;
    }

    const idx = pts.len / 2;
    const median = pts[pts.len/2];

    var node = try a.create(KDNode);
    node.l = try buildTree(a, pts[0..idx]);
    node.r = try buildTree(a, pts[idx+1..]);
    node.splt = median;
    node.vals = null;
    return node;
}

fn deleteTree(a: Allocator, node: *KDNode) void {
    if (node.l) |x| deleteTree(a, x);
    if (node.r) |x| deleteTree(a, x);
    a.destroy(node);
}

fn findNearest(root: *KDNode, pt: f32) Allocator.Error!f32 {
    var nearest_pt = root.splt.?;
    var best_dist = std.math.absFloat(pt-root.splt.?);
    var current = root;

    // descend down branches until we get to a leaf. keep track of nearest point at all times.
    while (true) {

        // we've made it to a leaf node. almost done.
        if (current.vals) |vals| {
            for (vals) |next_pt| {
                const d = std.math.absFloat(pt-next_pt);
                if (d<best_dist) {
                    nearest_pt = next_pt;
                    best_dist = d;
                }
            }
            return nearest_pt;
        }


        const splt_pt = current.splt.?;

        // test against split value
        const d = std.math.absFloat(pt-splt_pt);
        if (d<best_dist) {
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

fn printTree(a:Allocator,root:*KDNode, lvl:u8) Allocator.Error!void {

    var lvlstr = try a.alloc(u8, lvl*2);
    defer a.free(lvlstr);
    for (lvlstr) |*v| v.* = '-';
    print("{s}", .{lvlstr});

    if (root.vals) |vals| {
        print(" {d:.3} \n" , .{vals});
        return;
    }

    // we know splt is true
    const splt = root.splt.?;

    print(" {d:.3} \n" , .{splt});

    try printTree(a,root.l.?, lvl+1);
    try printTree(a,root.r.?, lvl+1);
}

test "kdnode test" {
    print("\n",.{});
    var a = allocator;

    var pts = try a.alloc(f32, 13);
    defer a.free(pts);
    for (pts) |*v| v.* = random.float(f32);

    std.sort.sort(f32, pts, {}, comptime std.sort.asc(f32));
    var q = try buildTree(a, pts[0..]);
    defer deleteTree(a, q);

    try printTree(a,q,0);
}



fn testNearest(a:Allocator,N:u32) !void {
    var pts = try a.alloc(f32, N);
    defer a.free(pts);
    for (pts) |*v| v.* = random.float(f32);

    std.sort.sort(f32, pts, {}, comptime std.sort.asc(f32));
    var q = try buildTree(a, pts[0..]);
    defer deleteTree(a, q);

    const nearest1 = try findNearest(q, pts[N/2]);
    print("pt = {} , nearest1 = {} \n", .{pts[N/2] , nearest1});
    try std.testing.expectEqual(pts[N/2], nearest1);

    const pt = random.float(f32);
    const nearest2 = try findNearest(q, pt);
    print("pt = {} , nearest2 = {} \n", .{pt , nearest2});
    try std.testing.expect(std.math.absFloat(pt - nearest2) < 10.0/@intToFloat(f32, N));
}

test "find nearest test" {
    print("\n",.{});
    var a = allocator;
    try testNearest(a, 3); // must be at least 3 or findNearest() fails
    try testNearest(a, 10);
    try testNearest(a, 100);
    try testNearest(a, 1000);
    try testNearest(a, 10000);

}










