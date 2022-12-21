const std = @import("std");
const print = std.debug.print;

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
    const alloc = std.testing.allocator;
    var pts = try alloc.alloc(f32, 100);
    defer alloc.free(pts);
    pts = alloc.resize(pts, pts.len + 1).?;
}

test "resize many times" {
    const alloc = std.testing.allocator;
    var count: u32 = 100;
    while (count < 10000) : (count += 100) {
        var pts = try alloc.alloc([2]f32, count);
        defer alloc.free(pts);

        for (pts) |*v, j| {
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

fn testresize(_pts: [][2]f32) !void {
    const alloc = std.testing.allocator;
    var pts = alloc.resize(_pts, _pts.len + 3).?;
    defer _ = alloc.shrink(pts, pts.len - 3);
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
    const V = @Vector(2,u32);
    const P = @Vector(2,f32);
    const x = V{3,4};
    _ = @intToFloat(P,x); // ERROR
    // _ = y;
}
