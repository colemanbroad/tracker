const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;

const trace = @import("trace");
const Span = trace.Span;
pub const enable_trace = true;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var allocator = gpa.allocator();
// var allocator = std.testing.allocator;
var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

const XY = enum {
    X,
    Y,
};

const V2 = @Vector(2, f32);
fn v2(x: [2]f32) V2 {
    return x;
}

const Pt = [2]f32;
const Volume = struct { min: Pt, max: Pt };

// Consider this as an alternative to the KDNode that avoids the issues with
// null
const NodeTag = enum { Split, Leaf, Empty };
const NodeUnion = union(NodeTag) {
    Split: struct { pt: Pt, l: *NodeUnion, r: *NodeUnion, dim: u8, vol: Volume },
    Leaf: struct { pt: Pt, vol: Volume },
    Empty: struct { vol: Volume },
};

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

fn ltPtsDims(dim: XY, lhs: Pt, rhs: Pt) bool {
    switch (dim) {
        .X => return lhs[0] < rhs[0],
        .Y => return lhs[1] < rhs[1],
    }
}

fn ltPtsIdx(idx: u8, lhs: Pt, rhs: Pt) bool {
    return lhs[idx] < rhs[idx];
}

fn ltPtX(_: anytype, lhs: Pt, rhs: Pt) bool {
    return lhs[0] < rhs[0];
}
fn ltPtY(_: anytype, lhs: Pt, rhs: Pt) bool {
    return lhs[1] < rhs[1];
}

fn dist(a: Pt, b: Pt) f32 {
    // return std.math.absFloat(a - b);
    const d = Pt{ a[0] - b[0], a[1] - b[1] };
    const norml2 = @sqrt(d[0] * d[0] + d[1] * d[1]);
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

fn printNode(n: NodeUnion) void {
    const at = std.meta.activeTag;

    switch (n) {
        .Empty => |_| print("Empty \n", .{}),
        .Leaf => |l| print("Leaf {d} \n", .{l.pt}),
        .Split => |s| {
            print("Split {d:0.3} {s} {s} \n", .{ s.pt, @tagName(at(s.l.*)), @tagName(at(s.r.*)) });
        },
    }
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
    const idx = pt_argmax(v2(span.max) - v2(span.min));
    std.sort.sort(Pt, pts, idx, ltPtsIdx);
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
fn pt_argmax(v: Pt) u8 {
    const m = @max(v[0], v[1]);
    if (v[0] == m) return 0;
    return 1;
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

        // var img = try im.Img2D([4]u8).init(nx, ny);

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

        res.pix.img[3 * nx + 50] = .{ 255, 255, 255, 255 };
        res.pix.img[3 * nx + 51] = .{ 255, 255, 255, 255 };
        res.pix.img[3 * nx + 52] = .{ 255, 255, 255, 255 };
        res.pix.img[3 * nx + 53] = .{ 255, 255, 255, 255 };

        return res;
    }

    fn update(this: This) !void {
        // _ = cc.SDL_LockSurface(this.surface);
        // for (this.pix.img, 0..) |v, i| {
        //     this.pix.img[i] = v;
        // }

        // for (this.pix.img, 0..) |p, i| {
        //     this.surface.pixels[i] = p;
        // }

        // this.surface.pixels = &this.pix.img[0];
        // this.setPixel(x: c_int, y: c_int, pixel: [4]u8)
        // cc.SDL_UnlockSurface(this.surface);

        const err = cc.SDL_UpdateWindowSurface(this.sdl_window);
        if (err != 0) {
            cc.SDL_Log("Error updating window surface: %s", cc.SDL_GetError());
            return error.SDLUpdateWindowFailed;
        }
    }

    fn setPixel(this: *This, x: c_int, y: c_int, pixel: [4]u8) void {
        const target_pixel = @ptrToInt(this.surface.pixels) +
            @intCast(usize, y) * @intCast(usize, this.surface.pitch) +
            @intCast(usize, x) * 4;
        @intToPtr(*u32, target_pixel).* = @bitCast(u32, pixel);
    }

    fn setPixels(this: *This, buffer: [][4]u8) void {
        _ = cc.SDL_LockSurface(this.surface);
        for (buffer, 0..) |v, i| {
            this.pix.img[i] = v;
        }
        cc.SDL_UnlockSurface(this.surface);
    }

    fn markBounds(this: *This) void {
        _ = cc.SDL_LockSurface(win.surface);
        // TOP LEFT BLUE
        im.drawCircle([4]u8, this.pix, 0, 0, 13, .{ 255, 0, 0, 255 });
        // TOP RIGHT GREEN
        im.drawCircle([4]u8, this.pix, @intCast(i32, this.nx), 0, 13, .{ 0, 255, 0, 255 });
        // BOT LEFT RED
        im.drawCircle([4]u8, this.pix, 0, @intCast(i32, this.ny), 13, .{ 0, 0, 255, 255 });
        // BOT RIGHT WHITE
        im.drawCircle([4]u8, this.pix, @intCast(i32, this.nx), @intCast(i32, this.ny), 13, .{ 255, 255, 255, 255 });
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
const im = @import("image_base.zig");

fn randomColor() [4]u8 {
    return .{
        random.int(u8),
        random.int(u8),
        255,
        255,
    };
}

fn drawTee(root: *NodeUnion) !void {
    const ParentNode = struct { parent: ?[2]i32, node: *NodeUnion };
    var node_q: [2 * N]ParentNode = undefined;
    var h_idx: usize = 0;
    var t_idx: usize = 0;

    // const TupT = struct { i32, i32, [2][2]i32 };
    // var point_list: [100]TupT = undefined;
    // var point_list_idx: usize = 0;

    const Counts = struct { empty: u16 = 0, leaf: u16 = 0, split: u16 = 0 };
    var counts = Counts{};

    node_q[h_idx] = .{ .parent = null, .node = root };
    h_idx += 1;

    while (t_idx < h_idx) {

        // if (try waitForUserInput() == 0) return;

        // win.setPixels(win.pix.img);

        try win.update();

        // print("\nt_idx = {d} , h_idx = {d} \n\n", .{ t_idx, h_idx });
        const p_and_n = node_q[t_idx];
        t_idx += 1;
        const n = p_and_n.node.*;
        const parent_pt = p_and_n.parent;
        _ = parent_pt;
        // print("pop a {s} \n", .{@tagName(std.meta.activeTag(n))});

        if (n == .Empty) {
            counts.empty += 1;
            continue;
        }

        const n_pt = switch (n) {
            .Leaf => |l| l.pt,
            .Split => |l| l.pt,
            else => unreachable,
        };

        // _ = cc.SDL_LockSurface(win.surface);

        const x = @floatToInt(i32, n_pt[0] * 750 + 25);
        const y = @floatToInt(i32, n_pt[1] * 750 + 25);
        // printNode(n);
        // if (x == 192 and y == 572) @breakpoint();

        im.drawCircle([4]u8, win.pix, x, y, 3, .{ 255, 255, 255, 255 });
        // Line to parent
        // if (parent_pt) |p| {
        // im.drawLineInBounds([4]u8, win.pix, p[0], p[1], x, y, .{ 255, 128, 0, 255 });
        // }

        // Draw the Vertical / Horizontal split line
        if (n == .Split) {
            const bv = n.Split.vol;

            const line: [2][2]i32 = switch (n.Split.dim) {
                0 => .{
                    .{ x, @floatToInt(i32, bv.min[1] * 750 + 25) },
                    .{ x, @floatToInt(i32, bv.max[1] * 750 + 25) },
                },
                1 => .{
                    .{ @floatToInt(i32, bv.min[0] * 750 + 25), y },
                    .{ @floatToInt(i32, bv.max[0] * 750 + 25), y },
                },
                else => unreachable,
            };

            im.drawLineInBounds([4]u8, win.pix, line[0][0], line[0][1], line[1][0], line[1][1], randomColor());
            // point_list[point_list_idx] = .{ x, y, line };
            // point_list_idx += 1;
        }

        // cc.SDL_UnlockSurface(win.surface);

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

    // @breakpoint();
}

const NNReturn = struct { dist: f32, pt: Pt, node: NodeUnion };

pub fn findNearestNeib(tree_root: *NodeUnion, query_point: Pt) NNReturn {
    // TODO: be more efficient than 2*N here. shouldn't be nearly that many.
    var node_stack: [100]NodeUnion = undefined;
    var h_idx: u32 = 0;

    node_stack[h_idx] = tree_root.*;
    h_idx += 1;

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

    if (traversing_node == .Leaf) {
        const n = traversing_node.Leaf;
        const next_dist = dist(n.pt, query_point);
        if (next_dist < current_min_dist) {
            current_min_dist = next_dist;
            current_min_node = traversing_node;
            current_pt = n.pt;
        }
    }

    while (h_idx > 0) {
        itercount += 1;
        // WARN: remember that h_idx points to the next empty spot (insert position)
        // but the last full position is at the previous index!
        const next_node = node_stack[h_idx - 1];
        h_idx -= 1;

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
        if (!(orthogonal_distance > current_min_dist)) {
            // add s.r
            node_stack[h_idx] = s.r.*;
            h_idx += 1;
        }
        if (!(orthogonal_distance < -current_min_dist)) {
            // add s.l
            node_stack[h_idx] = s.l.*;
            h_idx += 1;
        }
    }

    print("itercount = {d}\n", .{itercount});
    return .{ .dist = current_min_dist, .pt = current_pt, .node = current_min_node };
}

test "simple union experiment" {
    var a: NodeUnion = undefined;

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

test "build a Tree" {
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

const N = 10_001;

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
    win.markBounds();

    const a = allocator;
    var pts = try a.alloc(Pt, N);
    defer a.free(pts);
    for (pts) |*v| v.* = .{ random.float(f32), random.float(f32) };

    for (pts) |p| {
        const x = @floatToInt(i32, p[0] * 750 + 25);
        const y = @floatToInt(i32, p[1] * 750 + 25);
        im.drawCircle([4]u8, win.pix, x, y, 3, .{ 255, 255, 255, 255 });
    }
    try win.update();

    // std.sort.sort(f32, pts, {}, comptime std.sort.asc(f32));
    var tree_root = try buildTree(pts[0..], .{ .min = .{ 0, 0 }, .max = .{ 1, 1 } });

    // defer deleteNodeAndChildren(a, tree_root);
    // const testimg = try im.Img2D([4]u8).init(1000, 800);
    // win.

    var e: cc.SDL_Event = undefined;
    _ = cc.SDL_PollEvent(&e);

    // try drawTee(tree_root);

    for (0..1000) |_| {
        const query_point = Pt{ random.float(f32), random.float(f32) };

        const span_kdtree = trace.Span.open("kdtree");
        const nearest_neib = findNearestNeib(tree_root, query_point);
        _ = nearest_neib;
        span_kdtree.close();

        const span_brute_force = trace.Span.open("brute_force");
        var closest_pt: Pt = pts[0];
        var closest_dist: f32 = dist(query_point, pts[0]);
        for (pts) |p| {
            const d = dist(query_point, p);
            if (d < closest_dist) {
                closest_pt = p;
                closest_dist = d;
            }
        }
        span_brute_force.close(); // Span is closed automatically when the function returns

    }

    // print(
    //     "query {d:0.3}, nearest {d:0.3}, closest {d:0.3} \n\n",
    //     .{ query_point, nearest_neib.pt, closest_pt },
    // );

    // print(
    //     "nn.dist {d:0.3}, true dist {d:0.3} \n\n",
    //     .{ nearest_neib.dist, closest_dist },
    // );

    return 0;
}
