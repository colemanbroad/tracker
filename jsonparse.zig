const std = @import("std");

const TrackedCell = struct { time: i32, id: i32, pt: [2]u16, parent_id: [2]i32, isbi_id: i32 };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const al = gpa.allocator();

    const s = @embedFile("tracked_cells.json");
    const parsedData = try std.json.parseFromSlice([]TrackedCell, al, s[0..], .{});
    defer parsedData.deinit();

    for (parsedData.value, 0..) |row, i| {
        std.debug.print("Row {} is {any}\n", .{ i, row });
    }
}
