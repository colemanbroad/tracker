usingnamespace @import("image_base.zig");

const test_home = "/Users/broaddus/Desktop/work/zig-tracker/test-artifacts/imageBase/";

test "imageBase. saveU8AsTGAGrey" {
    print("\n", .{});
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // var grey = .{0}**(2^10);
    var grey = std.mem.zeroes([1 << 10]u8);
    for (grey) |*v, i| v.* = @intCast(u8, i % 256);
    print("\n number ;;; {} \n", .{1 << 5});
    try saveU8AsTGAGrey(&grey, 1 << 5, 1 << 5, test_home ++ "correct.tga");
    print("\n", .{});
}

test "imageBase. saveU8AsTGAGrey (h & w too small)" {
    print("\n", .{});
    // var grey = .{0}**(2^10);
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var grey = std.mem.zeroes([1 << 10]u8);
    for (grey) |*v, i| v.* = @intCast(u8, i % 256);
    print("\n number ;;; {} \n", .{1 << 6});
    try saveU8AsTGAGrey(&grey, 1 << 5, 1 << 4, test_home ++ "height_width_too_small.tga");
    print("\n", .{});
}

test "imageBase. saveU8AsTGAGrey (h & w too big)" {
    print("\n", .{});
    // var grey = .{0}**(2^10);
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var grey = std.mem.zeroes([1 << 10]u8);
    for (grey) |*v, i| v.* = @intCast(u8, i % 256);
    print("\n number ;;; {} \n", .{1 << 6});
    try saveU8AsTGAGrey(&grey, 1 << 5, 1 << 6, test_home ++ "height_width_too_big.tga");
    print("\n", .{});
}

test "imageBase. saveU8AsTGA" {
    print("\n", .{});
    var rgba = std.mem.zeroes([(1 << 10) * 4]u8);
    for (rgba) |*v, i| v.* = bl: {
        const x = switch (i % 4) {
            0 => i % 255, // red
            1 => (2 * i) % 255, // blue
            2 => (3 * i) % 255, // green
            3 => 255, // alpha
            else => unreachable,
        };
        break :bl @intCast(u8, x);
    };
    // const x = @intCast(u8, i % 256); // alpha channel changes too!
    try saveU8AsTGA(&rgba, 1 << 5, 1 << 5, test_home ++ "multicolor.tga");
    print("\n", .{});
}
