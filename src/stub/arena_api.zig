const std = @import("std");
const print = std.debug.print;

test {
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var al = arena.allocator();

    const p1 = try newpts(al, 100);
    const p2 = try newpts(al, 200);
    const p3 = try newpts(al, 300);

    print("{d}\n", .{p1});
    print("{d}\n", .{p2});
    print("{d}\n", .{p3});
}

fn newpts(al: std.mem.Allocator, size: u32) ![][2]f32 {
    var pts = try al.alloc([2]f32, size);
    for (pts, 0..) |*p, i| p.* = .{ @intToFloat(f32, i), @intToFloat(f32, i * i) };
    return pts;
}

// var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
// defer arena.deinit();
