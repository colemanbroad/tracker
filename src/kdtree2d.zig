const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;

// const trace = @import("tracy.zig").trace;
// const trace = @import("trace");
// pub const enable_trace = true;

// OK Here's my timer idea. Instead of writing to std.log on open and close
// it just saves the times in memory and writes them all at the end. This
// should allow for nanosecond level profiling.
// The interface should be very similar and easy. We pass in a SourceLocation
// and automatically get the function name.
// const trace = struct {
//     // var spanlist = std.ArrayList(trace.Span)

//     const Span = struct {
//         const OpenClose = struct {open:u64, close:u64};
//         const This = @This();

//         name: []const u8,
//         timings:[10_000]OpenClose,

//         pub fn open(this: This , comptime name : []const u8) void {

//         }
//     };
// };

var timer: std.time.Timer = undefined;

const Trace = struct {
    v: [2 * Ntrials]u64 = undefined,
    idx: usize = 0,

    pub fn tic(this: *Trace) void {
        this.v[this.idx] = timer.read();
        // this.v[this.idx] = std.time.nanoTimestamp();
        this.idx += 1;
    }
};

const traces = struct {
    var kdtree = Trace{};
    var brute = Trace{};
    var sorted = Trace{};
};

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

// Waits for user input on the command line. "Enter" sends input.
fn awaitCmdLineInput() !i64 {
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

    must_quit: bool = false,
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
        // print("CreateWindow [{}ms]\n", .{t2 - t1});

        t1 = milliTimestamp();
        const surface = SDL_GetWindowSurface(window) orelse {
            cc.SDL_Log("Unable to get window surface: %s", cc.SDL_GetError());
            return error.SDLInitializationFailed;
        };
        t2 = milliTimestamp();
        // print("SDL_GetWindowSurface [{}ms]\n", .{t2 - t1});

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
        // _ = cc.SDL_LockSurface(this.surface);
        // TOP LEFT BLUE
        im.drawCircle([4]u8, this.pix, 0, 0, 13, .{ 255, 0, 0, 255 });
        // TOP RIGHT GREEN
        im.drawCircle([4]u8, this.pix, @intCast(i32, this.nx), 0, 13, .{ 0, 255, 0, 255 });
        // BOT LEFT RED
        im.drawCircle([4]u8, this.pix, 0, @intCast(i32, this.ny), 13, .{ 0, 0, 255, 255 });
        // BOT RIGHT WHITE
        im.drawCircle([4]u8, this.pix, @intCast(i32, this.nx), @intCast(i32, this.ny), 13, .{ 255, 255, 255, 255 });
        // cc.SDL_UnlockSurface(this.surface);
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
var sdl_event: cc.SDL_Event = undefined;

const im = @import("image_base.zig");

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

        // if (try awaitCmdLineInput() == 0) return;
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

pub fn findNearestNeibKDTree(tree_root: *NodeUnion, query_point: Pt) NNReturn {
    // TODO: be more efficient than 2*N here. shouldn't be nearly that many.

    // const tracy3 = trace(@src());
    // defer tracy3.end();
    // const tracy = trace.Span.open(@src().fn_name);
    // defer tracy.close();
    traces.kdtree.tic();
    defer traces.kdtree.tic();

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
    var node_stack: [100]NodeUnion = undefined;
    var h_idx: u32 = 0;

    node_stack[h_idx] = tree_root.*;
    h_idx += 1;

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

        // Walk down the tree in the direction of query_point
        // If you're far away from the query point you can completly avoid some branches
        // If you're within the query point current best radius then you first search
        // down the closer side by adding it 2nd in the stack.
        if (orthogonal_distance < -current_min_dist) {
            node_stack[h_idx] = s.r.*;
            h_idx += 1;
        } else if (orthogonal_distance < 0) {
            node_stack[h_idx] = s.l.*;
            h_idx += 1;
            node_stack[h_idx] = s.r.*;
            h_idx += 1;
        } else if (orthogonal_distance < current_min_dist) {
            node_stack[h_idx] = s.r.*;
            h_idx += 1;
            node_stack[h_idx] = s.l.*;
            h_idx += 1;
        } else {
            node_stack[h_idx] = s.l.*;
            h_idx += 1;
        }

        // if (!(orthogonal_distance > current_min_dist)) {
        //     // add s.r
        //     node_stack[h_idx] = s.r.*;
        //     h_idx += 1;
        // }
        // if (!(orthogonal_distance < -current_min_dist)) {
        //     // add s.l
        //     node_stack[h_idx] = s.l.*;
        //     h_idx += 1;
        // }
    }

    // print("itercount = {d}\n", .{itercount});
    return .{ .dist = current_min_dist, .pt = current_pt, .node = current_min_node };
}

pub fn greatestLowerBoundIndex(pts: []Pt, dim: u8, query_point: Pt) !usize {
    // const tracy4 = trace(@src());
    // defer tracy4.end();
    // const tracy = trace.Span.open(@src().fn_name);
    // defer tracy.close();

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

// An event loop that only moves forward on KeyDown.
// Use this in combination with drawing to the window to visually step
// through algorithms!
pub fn awaitKeyPressAndUpdateWindow() void {
    if (win.must_quit) return;
    while (true) {
        _ = cc.SDL_WaitEvent(&sdl_event);
        if (sdl_event.type == cc.SDL_KEYDOWN) {
            switch (sdl_event.key.keysym.sym) {
                cc.SDLK_q => std.os.exit(0),
                cc.SDLK_f => {
                    win.must_quit = true;
                }, // finish
                else => {
                    win.update() catch {};
                    break;
                },
            }
        }
    }
}

pub fn findNearestNeibFromSortedList(pts: []Pt, query_point: Pt) Pt {
    // const tracy1 = trace(@src());
    // defer tracy1.end();
    // const tracy = trace.Span.open(@src().fn_name);
    // defer tracy.close();
    traces.sorted.tic();
    defer traces.sorted.tic();

    // First we find the greatest lower bound (index)
    const dim_idx = 0;
    const glb_idx = greatestLowerBoundIndex(pts, dim_idx, query_point) catch 0;

    im.drawCircle(
        [4]u8,
        win.pix,
        @floatToInt(i32, query_point[0] * 750 + 25),
        @floatToInt(i32, query_point[1] * 750 + 25),
        3,
        .{ 0, 0, 255, 255 },
    );

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

        im.drawLineInBounds(
            [4]u8,
            win.pix,
            @floatToInt(i32, query_point[0] * 750 + 25),
            @floatToInt(i32, query_point[1] * 750 + 25),
            @floatToInt(i32, current_pt[0] * 750 + 25),
            @floatToInt(i32, current_pt[1] * 750 + 25),
            color,
        );

        awaitKeyPressAndUpdateWindow();

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

        im.drawLineInBounds(
            [4]u8,
            win.pix,
            @floatToInt(i32, query_point[0] * 750 + 25),
            @floatToInt(i32, query_point[1] * 750 + 25),
            @floatToInt(i32, current_pt[0] * 750 + 25),
            @floatToInt(i32, current_pt[1] * 750 + 25),
            color,
        );

        awaitKeyPressAndUpdateWindow();

        idx_offset += 1;
    }

    return pts[current_best_idx];
    // return .{ .pt = pts[current_best_idx], .dist = current_best_dist };
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

const N = 101;
const Ntrials = 5_000;

pub fn findNearestNeibBruteForce(pts: []Pt, query_point: Pt) Pt {
    // const tracy2 = trace(@src());
    // defer tracy2.end();

    // const tracy = trace.Span.open(@src().fn_name);
    // defer tracy.close();

    traces.brute.tic();
    defer traces.brute.tic();

    var closest_pt: Pt = pts[0];
    var closest_dist: f32 = dist(query_point, pts[0]);
    for (pts) |p| {
        // me too
        const d = dist(query_point, p);

        // test me out
        if (d < closest_dist) {
            closest_pt = p;
            closest_dist = d;
        }
    }
    return closest_pt;
}

pub fn main() !u8 {
    timer = try std.time.Timer.start();

    // Create random points
    const a = allocator;
    var pts = try a.alloc(Pt, N);
    defer a.free(pts);
    for (pts) |*v| v.* = .{ random.float(f32), random.float(f32) };

    // Initialize Drawing Pallete
    //
    if (cc.SDL_Init(cc.SDL_INIT_VIDEO) != 0) return error.SDLInitializationFailed;
    defer cc.SDL_Quit();
    win = try Window.init(1000, 800);
    // win.markBounds();
    for (pts) |p| {
        const x = @floatToInt(i32, p[0] * 750 + 25);
        const y = @floatToInt(i32, p[1] * 750 + 25);
        im.drawCircle([4]u8, win.pix, x, y, 3, .{ 255, 255, 255, 255 });
    }
    try win.update();
    // _ = cc.SDL_PollEvent(&sdl_event);
    // win.must_quit = true;

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
        const nn_brute_force = findNearestNeibBruteForce(pts, query_point);
        // if (query_point[0] == 0.0855857804) @breakpoint();
        const nn_bsorted = findNearestNeibFromSortedList(pts, query_point);

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

    const stat_kdtree = analyze(traces.kdtree);
    const stat_brute = analyze(traces.brute);
    const stat_sorted = analyze(traces.sorted);

    print("{s:20} mean {d:>12.0} [ns]   stddev {d:>12.0} \n", .{ "kdtree", stat_kdtree.mean, stat_kdtree.stddev });
    print("{s:20} mean {d:>12.0} [ns]   stddev {d:>12.0} \n", .{ "brute", stat_brute.mean, stat_brute.stddev });
    print("{s:20} mean {d:>12.0} [ns]   stddev {d:>12.0} \n", .{ "sorted", stat_sorted.mean, stat_sorted.stddev });

    print("\n\n\n", .{});

    print("{any}\n", .{stat_kdtree});
    print("{any}\n", .{stat_brute});
    print("{any}\n", .{stat_sorted});

    return 0;
}

fn analyze(trace: Trace) Stats {
    var tics = blk: {
        var tics: [Ntrials]f32 = undefined;
        for (&tics, 0..) |*t, i| {
            t.* = @intToFloat(f32, trace.v[2 * i + 1] - trace.v[2 * i]);
        }
        break :blk tics;
    };

    const s = statistics(f32, &tics);
    return s;
}

const Stats = struct {
    mean: f64,
    min: f64,
    max: f64,
    mode: f64,
    median: f64,
    stddev: f64,

    pub fn format(self: Stats, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        try writer.print("{s:9} {d:>10.0}\n", .{ "mean", self.mean });
        try writer.print("{s:9} {d:>10.0}\n", .{ "median", self.median });
        try writer.print("{s:9} {d:>10.0}\n", .{ "mode", self.mode });
        try writer.print("{s:9} {d:>10.0}\n", .{ "min", self.min });
        try writer.print("{s:9} {d:>10.0}\n", .{ "max", self.max });
        try writer.print("{s:9} {d:>10.0}\n", .{ "std dev", self.stddev });

        try writer.writeAll("");
    }
};
fn statistics(comptime T: type, arr: []T) Stats {

    // var s = Stats{
    //     .mean=0,
    //     .stddev = 0,
    //     .min=arr[0],
    //     .max=arr[0],
    //     .mode=arr[0],
    //     .median=arr[0],
    // };

    var s: Stats = undefined;

    s.mean = 0;
    s.stddev = 0;

    std.sort.heap(T, arr, {}, std.sort.asc(T));
    for (arr) |x| {
        s.mean += x;
        s.stddev += x * x;
        // s.min = @min(s.min, x);
        // s.max = @max(s.max, x);
    }
    s.mean /= @intToFloat(f32, arr.len);
    s.stddev /= @intToFloat(f32, arr.len);
    s.stddev -= s.mean * s.mean;
    s.stddev = @sqrt(s.stddev);

    // @breakpoint();
    s.min = arr[0];
    s.max = arr[arr.len - 1];
    s.median = arr[arr.len / 2];

    var max_count: u32 = 0;
    var current_count: u32 = 1;
    s.mode = arr[0];
    for (1..arr.len) |i| { // exclusive on right ?
        if (arr[i] != arr[i - 1]) {
            current_count = 1;
            continue;
        }

        current_count += 1;
        if (current_count <= max_count) continue;

        max_count = current_count;
        s.mode = arr[i];
    }

    return s;
}
