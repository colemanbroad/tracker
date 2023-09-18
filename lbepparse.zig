const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const al = gpa.allocator();
    _ = al;

    // load("")

    // fn waitForUserInput() !i64 {
    //     if (@import("builtin").is_test) return 0;

    //     const stdin = std.io.getStdIn().reader();
    //     const stdout = std.io.getStdOut().writer();

    //     var buf: [10]u8 = undefined;

    //     try stdout.print("Press 0 to quit: ", .{});

    //     if (try stdin.readUntilDelimiterOrEof(buf[0..], '\n')) |user_input| {
    //         const res = std.fmt.parseInt(i64, user_input, 10) catch return 1;
    //         if (res == 0) return 0;
    //     }
    //     return 1;
    // }

}

pub fn load(name: []const u8) !void {
    const file = try std.fs.cwd().openFile(name, .{});
    defer file.close();
    const reader = file.reader();

    volume_loop_max_index = try reader.readIntLittle(usize);
    for (&volume_loop_indices) |*v| v.* = try reader.readIntLittle(usize);
    _ = try reader.readAll(std.mem.sliceAsBytes(&volume_loop_mem));

    // return error.Success;
    // return true;
    // try writer.writeIntLittle(usize,temp_screen_loop_len);
    // try writer.writeAll(std.mem.sliceAsBytes(&volume_loop_indices));
    // try writer.writeAll(std.mem.sliceAsBytes(&volume_loop_mem));
}
