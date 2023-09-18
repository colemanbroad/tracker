const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const al = gpa.allocator();

    // const ParentID = enum {
    //     empty,
    // };
    // const Foo = struct { a: i32, b: bool, c: []u8 };
    // const Foo = struct { time: i32, id: i32, pt: [2]u16, parent_id: union(ParentID) {
    // empty: i8,
    // }, isbi_id: i32 };
    const Foo = struct { time: i32, id: i32, pt: [2]u16, parent_id: [2]i32, isbi_id: i32 };

    // const file = try std.fs.cwd().openFile(name, .{});
    // defer file.close();
    // const reader = file.reader();

    // volume_loop_max_index = try reader.readIntLittle(usize);
    // for (&volume_loop_indices) |*v| v.* = try reader.readIntLittle(usize);
    // _ = try reader.readAll(std.mem.sliceAsBytes(&volume_loop_mem));

    const s = @embedFile("celltracks.json");
    const parsedData = try std.json.parseFromSlice([]Foo, al, s[0..], .{});
    defer parsedData.deinit();

    // std.debug.print("Parsed JSON is {any}\n\n", .{parsedData.value});
    for (parsedData.value, 0..) |row, i| {
        std.debug.print("Row {} is {any}\n", .{ i, row });
    }

    // const A = enum { a, b };
    // const B = union(A) { a: i8, b: [2]u32 };
    // const C = struct { alpha: B, beta: i8 };
    // const s =
    //     \\[ {"alpha": -3, "beta": 3}, {"alpha": [1,5], "beta": 3}]
    // ;
    // const parsedData = try std.json.parseFromSlice([2]C, al, s[0..], .{});
    // defer parsedData.deinit();
    // std.debug.print("Parsed JSON is {any}\n\n", .{parsedData.value});

    // const Foo = O
    // const s =
    //     \\ {
    //     \\   "a": 15, "b": true,
    //     \\   "c": "hello world"
    //     \\ }
    // ;

    // const s1 =
    //     \\{"time": 0, "id": 0, "pt": [718, 690], "parent_id": -1, "isbi_id": 1}
    // ;
    // const s2 =
    //     \\[{"time": 0, "id": 0, "pt": [718, 690], "parent_id": -1, "isbi_id": 1},
    //     \\  {"time": 0, "id": 1, "pt": [210, 594], "parent_id": -1, "isbi_id": 2},
    //     \\  {"time": 0, "id": 2, "pt": [356, 418], "parent_id": -1, "isbi_id": 3},
    //     \\  {"time": 0, "id": 3, "pt": [270, 514], "parent_id": -1, "isbi_id": 4},
    //     \\  {"time": 0, "id": 4, "pt": [398, 642], "parent_id": -1, "isbi_id": 5},
    //     \\  {"time": 0, "id": 5, "pt": [362, 510], "parent_id": -1, "isbi_id": 6},
    //     \\  {"time": 0, "id": 6, "pt": [278, 242], "parent_id": -1, "isbi_id": 7},
    //     \\  {"time": 0, "id": 7, "pt": [344, 594], "parent_id": -1, "isbi_id": 8},
    //     \\  {"time": 0, "id": 8, "pt": [478, 482], "parent_id": -1, "isbi_id": 9},
    //     \\  {"time": 0, "id": 9, "pt": [444, 576], "parent_id": -1, "isbi_id": 10},
    //     \\  {"time": 0, "id": 10, "pt": [588, 722], "parent_id": -1, "isbi_id": 11},
    //     \\  {"time": 0, "id": 11, "pt": [606, 656], "parent_id": -1, "isbi_id": 12},
    //     \\  {"time": 0, "id": 12, "pt": [730, 606], "parent_id": -1, "isbi_id": 13},
    //     \\  {"time": 0, "id": 13, "pt": [688, 554], "parent_id": -1, "isbi_id": 14},
    //     \\  {"time": 0, "id": 14, "pt": [784, 508], "parent_id": -1, "isbi_id": 15},
    //     \\  {"time": 1, "id": 0, "pt": [206, 598], "parent_id": [0, 1], "isbi_id": 2},
    //     \\  {"time": 1, "id": 1, "pt": [248, 480], "parent_id": [0, 3], "isbi_id": 4},
    //     \\  {"time": 1, "id": 2, "pt": [402, 648], "parent_id": [0, 4], "isbi_id": 5},
    //     \\  {"time": 1, "id": 3, "pt": [370, 408], "parent_id": [0, 2], "isbi_id": 3},
    //     \\  {"time": 1, "id": 4, "pt": [736, 700], "parent_id": [0, 0], "isbi_id": 1},
    //     \\  {"time": 1, "id": 5, "pt": [252, 264], "parent_id": [0, 6], "isbi_id": 7},
    //     \\  {"time": 1, "id": 6, "pt": [352, 510], "parent_id": [0, 5], "isbi_id": 6},
    //     \\  {"time": 1, "id": 7, "pt": [344, 590], "parent_id": [0, 7], "isbi_id": 8},
    //     \\  {"time": 1, "id": 8, "pt": [438, 576], "parent_id": [0, 9], "isbi_id": 10},
    //     \\  {"time": 1, "id": 9, "pt": [476, 472], "parent_id": [0, 8], "isbi_id": 9},
    //     \\  {"time": 1, "id": 10, "pt": [692, 554], "parent_id": [0, 13], "isbi_id": 14},
    //     \\  {"time": 1, "id": 11, "pt": [732, 608], "parent_id": [0, 12], "isbi_id": 13},
    //     \\  {"time": 1, "id": 12, "pt": [590, 724], "parent_id": [0, 10], "isbi_id": 11},
    //     \\  {"time": 1, "id": 13, "pt": [600, 664], "parent_id": [0, 11], "isbi_id": 12},
    //     \\  {"time": 1, "id": 14, "pt": [960, 586], "parent_id": -1, "isbi_id": 16}]
    // ;
    // _ = s2;
    // // @breakpoint();
}
