/// A tree that automatically splits 2D points up into vertical and horizontal segments.
///
///
/// TODO:
/// - split pts into LEFT PTS < PiVoT <= RIGHT PTS with a single pass. use std.ArrayList
/// - how to determine pivot? find median by sorting points. then split.
///
///
/// still need to do: be able to add points and detect intersection with BBoxes.
/// to construct a tree i pass in a list of BBoxes and get back a tree.
/// i can query the tree with a new point / bbox and determine containment / overlap.
/// i can add new bboxes to the tree.
/// Q: is it _enough_ to determine bbox containment for the purpose of delaunay construction?
///    The bounding box of a triangle is is not contained within the bounding circle (or vice versa).
///    To find a tri that contains pt p, we can still construct a tree that branches tris by centroid location,
///    then once we've found tri with closes centroid location to `p` we can search backwards through tree.
///    We only need to find one hit for Delaunay. Then we can do the remaining search in the triangle graph (voronoi graph).
/// return the root node
/// split triangles into equally balanced piles.
/// splits alternate between vertical and horizontal ? or do we always split both ways?
/// compute point for each tri, e.g. midpoint.
///
const std = @import("std");
// const geo = @import("geometry.zig");

const print = std.debug.print;

const Allocator = std.mem.Allocator;
// const Vec2 = geo.Vec2;
// const Tri = @import("delaunay.zig").Tri;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

var allocator = gpa.allocator();
// var allocator = std.testing.allocator;
var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

const XY = enum {
    X,
    Y,
};

const Pt = [2]f32;

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

fn ltPtX(_: anytype, lhs: Pt, rhs: Pt) bool {
    return lhs[0] < rhs[0];
}
fn ltPtY(_: anytype, lhs: Pt, rhs: Pt) bool {
    return lhs[1] < rhs[1];
}

/// Some better ideas. Don't recurse. Just alloc some memory for indices and
/// have a scheme
/// TODO: WIP
fn buildTreeNoRecurse(a: Allocator, pts: []Pt) Allocator.Error!void {
    _ = a;
    const xsorted = std.sort.sort(Pt, pts, 0, ltPtX);
    _ = xsorted;
    const ysorted = std.sort.sort(Pt, pts, 0, ltPtY);
    _ = ysorted;

    // We don't want to hold nodes to build the tree, we need
}

// IDEA merges vals and splt into union ?
const KDNode = struct {
    l: ?*KDNode = null, // less than pivot
    r: ?*KDNode = null, // greater than pivot
    dim: XY, // just ignore it if we're at a leaf
    splt: Pt,
    vals: [3]?Pt = [1]?Pt{null} ** 3,
};

// The simplest possible tree builder. Alternate X/Y splits (not max variance
// splits).
fn buildTree(a: Allocator, pts: []Pt, dim: XY) Allocator.Error!*KDNode {
    if (pts.len <= 4) {
        // Returns pointer to UNDEFINED memory. Allocation is not initialization! No RAII.
        var node = try a.create(KDNode);
        for (pts[1..], 0..) |p, i| node.vals[i] = p;
        node.splt = pts[0];
        node.dim = dim; // doesn't matter
        // @breakpoint();
        node.vals = [1]?Pt{null} ** 3;
        node.l = null;
        node.r = null;
        return node;
    }

    // sort by dimension
    std.sort.sort(Pt, pts, dim, ltPtsDims);

    const idx: usize = pts.len / 2;
    const median = pts[idx];

    var node = try a.create(KDNode);
    const next_split = switch (dim) {
        .X => XY.Y,
        .Y => XY.X,
    };
    node.l = try buildTree(a, pts[0..idx], next_split);
    node.r = try buildTree(a, pts[idx + 1 ..], next_split);
    node.splt = median;
    node.dim = dim;
    node.vals = [1]?Pt{null} ** 3;
    return node;
}

fn deleteNodeAndChildren(a: Allocator, node: *KDNode) void {
    // @breakpoint();
    if (node.l) |x| deleteNodeAndChildren(a, x);
    if (node.r) |x| deleteNodeAndChildren(a, x);
    a.destroy(node);
}

fn dist(a: Pt, b: Pt) f32 {
    // return std.math.absFloat(a - b);
    const d = Pt{ a[0] - b[0], a[1] - b[1] };
    const norml2 = @sqrt(d[0] * d[0] + d[1] * d[1]);
    return norml2;
}

// Given a point and a tree, find the nearest (l2dist) point in the tree.
fn findNearest(root: *KDNode, pt_query: Pt) Allocator.Error!Pt {
    var nearest_pt = root.splt;
    var best_dist = dist(pt_query, root.splt);
    var current = root;

    while (true) {

        // we've made it to a leaf node. almost done.
        if (current.vals) |vals| {
            for (vals) |next_pt| {
                const d = dist(pt_query, next_pt);
                if (d < best_dist) {
                    nearest_pt = next_pt;
                    best_dist = d;
                }
            }
            return nearest_pt;
        }

        const splt_pt = current.splt.?;

        // test against split value
        const d = dist(pt_query, splt_pt);
        if (d < best_dist) {
            nearest_pt = splt_pt;
            best_dist = d;
        }

        if (pt_query > splt_pt) {
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

    for (root.vals) |val| {
        if (val == null) continue;
        print(" {d:.3} \n", .{val.?});
        return;
    }

    print(" {} - {d:.3} \n", .{ root.dim, root.splt });

    if (root.l == null) return;

    try printTree(a, root.l.?, lvl + 1);
    try printTree(a, root.r.?, lvl + 1);
}

const im = @import("image_base.zig");
// const Img2D = im.Img2D;

fn drawTee(root: *KDNode) !void {
    var node_q: [300]*KDNode = undefined;
    var h_idx: usize = 0;
    var t_idx: usize = 0;
    node_q[h_idx] = root;
    h_idx += 1;

    while (t_idx < h_idx) {
        if (try waitForUserInput() == 0) return;

        win.setPixels(win.pix.img);
        try win.update();

        const n = node_q[t_idx];
        t_idx += 1;

        const x = @floatToInt(i32, n.splt[0] * 750 + 25);
        const y = @floatToInt(i32, n.splt[1] * 750 + 25);
        print("{any}", .{n});
        im.drawCircle([4]u8, win.pix, x, y, 3, .{ 255, 255, 255, 255 });

        if (n.l == null) continue;
        node_q[h_idx] = n.l.?;
        h_idx += 1;
        node_q[h_idx] = n.r.?;
        h_idx += 1;

        // im.drawCircle([4]u8, win.pix, x, y, 3, .{ 255, 255, 255, 255 });
        const xl0 = @floatToInt(i32, n.l.?.splt[0] * 750 + 25);
        const xl1 = @floatToInt(i32, n.l.?.splt[1] * 750 + 25);
        const xr0 = @floatToInt(i32, n.r.?.splt[0] * 750 + 25);
        const xr1 = @floatToInt(i32, n.r.?.splt[1] * 750 + 25);
        im.drawLineInBounds([4]u8, win.pix, xl0, xl1, x, y, .{ 255, 128, 0, 255 });
        im.drawLineInBounds([4]u8, win.pix, xr0, xr1, x, y, .{ 255, 128, 0, 255 });
    }
}

test "kdnode test" {
    print("\n", .{});
    var a = allocator;

    var pts = try a.alloc(Pt, 13);
    defer a.free(pts);
    for (pts) |*v| v.* = .{ random.float(f32), random.float(f32) };

    // std.sort.sort(f32, pts, {}, comptime std.sort.asc(f32));
    var q = try buildTree(a, pts[0..], .X);
    defer deleteNodeAndChildren(a, q);

    try printTree(a, q, 0);
}

fn testNearest(a: Allocator, N: u32) !void {
    var pts = try a.alloc(f32, N);
    defer a.free(pts);
    for (pts) |*v| v.* = random.float(f32);

    std.sort.sort(f32, pts, {}, comptime std.sort.asc(f32));
    var q = try buildTree(a, pts[0..], .X);
    defer deleteNodeAndChildren(a, q);

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

const cc = struct {
    pub usingnamespace @cImport({
        @cInclude("SDL2/SDL.h");
    });
};
const milliTimestamp = std.time.milliTimestamp;

const SDL_WINDOWPOS_UNDEFINED = @bitCast(c_int, cc.SDL_WINDOWPOS_UNDEFINED_MASK);

/// For some reason, this isn't parsed automatically. According to SDL docs, the
/// surface pointer returned is optional!
extern fn SDL_GetWindowSurface(window: *cc.SDL_Window) ?*cc.SDL_Surface;

const Window = struct {
    sdl_window: *cc.SDL_Window,
    surface: *cc.SDL_Surface,
    pix: im.Img2D([4]u8),

    needs_update: bool,
    update_count: u64,
    windowID: u32,
    nx: u32,
    ny: u32,

    const This = @This();

    /// WARNING: c managed heap memory mixed with our custom allocator
    fn init(nx: u32, ny: u32) !This {
        var t1: i64 = undefined;
        var t2: i64 = undefined;

        // window = SDL_CreateWindow( "SDL Tutorial", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN );

        t1 = milliTimestamp();
        const window = cc.SDL_CreateWindow(
            "Main Volume",
            SDL_WINDOWPOS_UNDEFINED,
            SDL_WINDOWPOS_UNDEFINED,
            @intCast(c_int, nx),
            @intCast(c_int, ny),
            // cc.SDL_WINDOW_OPENGL,
            cc.SDL_WINDOW_SHOWN,
        ) orelse {
            cc.SDL_Log("Unable to create window: %s", cc.SDL_GetError());
            return error.SDLInitializationFailed;
        };
        t2 = milliTimestamp();
        print("CreateWindow [{}ms]\n", .{t2 - t1});

        t1 = milliTimestamp();
        const surface = SDL_GetWindowSurface(window) orelse {
            cc.SDL_Log("Unable to get window surface: %s", cc.SDL_GetError());
            return error.SDLInitializationFailed;
        };
        t2 = milliTimestamp();
        print("SDL_GetWindowSurface [{}ms]\n", .{t2 - t1});

        // @breakpoint();
        var pix: [][4]u8 = undefined;
        pix.ptr = @ptrCast([*][4]u8, surface.pixels.?);
        pix.len = nx * ny;

        var img = im.Img2D([4]u8){
            .img = pix,
            .nx = nx,
            .ny = ny,
        };

        const res = .{
            // .pix = @ptrCast([*c][4]u8, surface.pixels.?),
            // .pix = pix,
            .sdl_window = window,
            .surface = surface,
            .pix = img,
            .needs_update = false,
            .update_count = 0,
            .windowID = cc.SDL_GetWindowID(window),
            .nx = nx,
            .ny = ny,
        };

        res.pix.img[50] = .{ 255, 255, 255, 255 };
        res.pix.img[51] = .{ 255, 255, 255, 255 };
        res.pix.img[52] = .{ 255, 255, 255, 255 };
        res.pix.img[53] = .{ 255, 255, 255, 255 };

        return res;
    }

    fn update(this: This) !void {
        const err = cc.SDL_UpdateWindowSurface(this.sdl_window);
        if (err != 0) {
            cc.SDL_Log("Error updating window surface: %s", cc.SDL_GetError());
            return error.SDLUpdateWindowFailed;
        }
    }

    // fn setPixel(this: *This, x: c_int, y: c_int, pixel: [4]u8) void {
    //     const target_pixel = @ptrToInt(this.surface.pixels) +
    //         @intCast(usize, y) * @intCast(usize, this.surface.pitch) +
    //         @intCast(usize, x) * 4;
    //     @intToPtr(*u32, target_pixel).* = @bitCast(u32, pixel);
    // }

    fn setPixels(this: *This, buffer: [][4]u8) void {
        _ = cc.SDL_LockSurface(this.surface);
        for (buffer, 0..) |v, i| {
            this.pix.img[i] = v;
        }
        cc.SDL_UnlockSurface(this.surface);
    }

    // fn setPixelsFromRectangle(this: *This, img: im.Img2D([4]u8), r: Rect) void {
    //     _ = cc.SDL_LockSurface(this.surface);

    //     const x_zoom = @intToFloat(f32, this.nx) / @intToFloat(f32, r.xmax - r.xmin);
    //     const y_zoom = @intToFloat(f32, this.ny) / @intToFloat(f32, r.ymax - r.ymin);

    //     for (this.pix.img, 0..) |*w, i| {
    //         const x_idx = r.xmin + divFloorIntByFloat(i % this.nx, x_zoom);
    //         const y_idx = r.ymin + divFloorIntByFloat(@divFloor(i, this.nx), y_zoom);
    //         const v = img.get(x_idx, y_idx).*;
    //         w.* = v;
    //     }
    //     cc.SDL_UnlockSurface(this.surface);
    // }
};

var win: Window = undefined;

pub fn main() !u8 {

    // Setup SDL & open window
    var t1 = milliTimestamp();
    if (cc.SDL_Init(cc.SDL_INIT_VIDEO) != 0) {
        cc.SDL_Log("Unable to initialize SDL: %s", cc.SDL_GetError());
        return error.SDLInitializationFailed;
    }
    defer cc.SDL_Quit();

    var t2 = milliTimestamp();
    print("SDL_Init [{}ms]\n", .{t2 - t1});

    win = try Window.init(1000, 800);
    // defer win.deinit();

    const a = allocator;
    var pts = try a.alloc(Pt, 101);
    defer a.free(pts);
    for (pts) |*v| v.* = .{ random.float(f32), random.float(f32) };

    // std.sort.sort(f32, pts, {}, comptime std.sort.asc(f32));
    var q = try buildTree(a, pts[0..], .X);
    // defer deleteNodeAndChildren(a, q);

    // const testimg = try im.Img2D([4]u8).init(1000, 800);

    // win.

    var e: cc.SDL_Event = undefined;
    _ = cc.SDL_PollEvent(&e);

    try drawTee(q);

    return 0;
    // win.setPixels(testimg.img);
    // try win.update();

    // var quit = false;
    // while (quit == false) {
    //     if (e.type == cc.SDL_QUIT) quit = true;

    //     for (win.pix.img) |*v| v.* = .{
    //         random.int(u8),
    //         random.int(u8),
    //         random.int(u8),
    //         255,
    //     };
    //     // win.setPixels(testimg.img);
    //     try win.update();
    // }

    // while (false) {
    //     // if (win.needs_update == false) {
    //     //     continue;
    //     // }
    //     // cc.SDL_Delay(16);

    //     for (testimg.img) |*v| v.* = .{
    //         random.int(u8),
    //         random.int(u8),
    //         random.int(u8),
    //         255,
    //     };
    //     win.setPixels(testimg.img);
    //     try win.update();

    //     const val = try waitForUserInput();
    //     if (val == 1) break;

    //     // @breakpoint();
    //     // if (val == 'q') break;

    //     win.update_count += 1;
    // }
}

test "test window creation" {
    const wind = try Window.init(1000, 800);
    try wind.update();
}
