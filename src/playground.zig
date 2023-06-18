const std = @import("std");
const print = std.debug.print;

// var gpa = std.heap.GeneralPurposeAllocator(.{}){};
// const alloc = gpa.allocator();
const alloc = std.heap.page_allocator;

fn print2(
    comptime src_info: std.builtin.SourceLocation,
    comptime fmt: []const u8,
    args: anytype,
) void {
    if (true) return;
    const s1 = comptime std.fmt.comptimePrint("{s}:{d}:{d} ", .{ src_info.file, src_info.line, src_info.column });
    std.debug.print(s1[41..] ++ fmt, args);
}

test "resize a slice" {
    var pts = try alloc.alloc(f32, 100);
    defer alloc.free(pts);
    const failed = alloc.resize(pts, pts.len + 1);
    _ = failed;
}

test "resize many times" {
    var count: u32 = 100;
    while (count < 10000) : (count += 100) {
        var pts = try alloc.alloc([2]f32, count);
        defer alloc.free(pts);

        for (pts, 0..) |*v, j| {
            const i = @intToFloat(f32, j);
            v.* = .{ i, i * i };
        }

        try testresize(pts);
        print2(
            @src(),
            "after resize...\npts = {d}\n",
            .{pts[pts.len - 3 ..]},
        );
        print2(
            @src(),
            "len={d}",
            .{pts.len},
        );
    }
}

fn testresize(pts: [][2]f32) !void {
    _ = alloc.resize(pts, pts.len + 3);
    defer _ = alloc.resize(pts, pts.len - 3);
    pts[pts.len - 3] = .{ 1, 0 };
    pts[pts.len - 2] = .{ 1, 1 };
    pts[pts.len - 1] = .{ 1, 2 };
    print2(
        @src(),
        "pts[-3..] = {d}\n",
        .{pts[pts.len - 3 ..]},
    );
    print2(
        @src(),
        "in testresize() : len={d}",
        .{pts.len},
    );
}

/// collapse repeated elements [1,1,1,2,2,3,3,1,1,4,4] → [1,2,3,1,4]
pub fn collapseAllRepeated(comptime T: type, mem: []T) []T {
    var write: u16 = 0;
    var read: u16 = 1;
    while (read < mem.len) : (read += 1) {
        if (mem[write] == mem[read]) continue;
        write += 1;
        mem[write] = mem[read];
    }
    return mem[0 .. write + 1];
}

test "float cast on @Vector" {
    const V = @Vector(2, u32);
    const P = @Vector(2, f32);
    _ = P;
    const x = V{ 3, 4 };
    _ = x;
    // _ = @intToFloat(P, x); // Compile Error
    // _ = y;
}

export fn add(a: i32, b: i32) i32 {
    return a + b;
}
export fn sum(a: [*]i32, n: u32) i32 {
    var tot: i32 = 0;
    var i: u32 = 0;
    while (i < n) {
        tot += a[i];
        i += 1;
        // print("Print {} me {} you fool!\n", .{tot,i});
    }
    return tot;
}

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

// Zig doesn't have tuple-of-types i.e. product types yet so all fields must be named. This is probably good.
// https://github.com/ziglang/zig/issues/4335

// Yes they do now!!
test "tuple of types exists ?" {
    const T = struct { u16, u8, bool };
    const t1 = T{ 14, 255, true };
    _ = t1;
}
