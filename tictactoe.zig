const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const assert = std.debug.assert;
const min = std.math.min;
const max = std.math.max;

var prng = std.rand.DefaultPrng.init(1);
const random = prng.random();
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Local imports
// const im = @import("image_base.zig");
// const Tracer = @import("tracelet.zig").Tracer;
// var tracer: Tracer(100) = undefined;
// Get an SDL window we can use for visualizing algorithms.
// const sdlw = @import("sdl-window.zig");
// var win: ?sdlw.Window = null;
// var win_plot: ?sdlw.Window = null;
// const nn_tools = @import("kdtree2d.zig");

fn PermutationIterator(comptime n: u8) type {
    return struct {
        const Self = @This();

        count: u32 = 0,
        i: u32 = 0,
        c: [n]u32,
        arr: [n]u32,

        fn init(arr: [n]u32) Self {
            return .{ .c = [1]u32{0} ** n, .arr = arr };
        }

        fn next(this: *Self) ?[n]u32 {
            var c = &this.c;
            var arr = &this.arr;
            const i = this.i;

            if (c[i] < i) {
                if (i % 2 == 0) {
                    // swap 0 , i
                    const tmp = arr[0];
                    arr[0] = arr[i];
                    arr[i] = tmp;
                } else {
                    // swap c[i] , i
                    const tmp = arr[c[i]];
                    arr[c[i]] = arr[i];
                    arr[i] = tmp;
                }

                // print("{d:10} == {d} \n", .{ this.count, arr.* });
                this.count += 1;

                c[i] += 1;
                this.i = 1;
            } else {
                c[i] = 0;
                this.i += 1;
            }
            if (this.i == n) return null;
            return arr.*;
        }
    };
}

fn enumeratePermutations(comptime n: u8, arr: [n]u32) void {
    var count: u32 = 0;

    // const n = arr.len;
    // var c = try allocator.alloc(u32, n);
    var c = [1]u32{0} ** n;
    // for (c) |*x| x.* = 0;

    print("{d:10} == {d} \n", .{ count, arr.* });
    count += 1;

    var i: u32 = 0;
    // i is the index into c. arr.len = c.len = n.
    while (i < n) {
        if (c[i] < i) {
            if (i % 2 == 0) {
                // swap 0 , i
                const tmp = arr[0];
                arr[0] = arr[i];
                arr[i] = tmp;
            } else {
                // swap c[i] , i
                const tmp = arr[c[i]];
                arr[c[i]] = arr[i];
                arr[i] = tmp;
            }

            print("{d:10} == {d} \n", .{ count, arr.* });
            count += 1;

            c[i] += 1;
            i = 1;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
}

// A maximin version of tictac toe
// const Player = enum { x, o };

pub fn main() !void {
    // var player = Player.x;
    const Res = struct { x: u32, o: u32, draw: u32 };
    var results = Res{ .x = 0, .o = 0, .draw = 0 };
    var allmoves = [9]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    var perms = PermutationIterator(9).init(allmoves);

    var i: u32 = 0;
    while (perms.next()) |p| {
        // for (enumeratePermutations(9, &allmoves), 0..) |p, i| {
        // the permutation defines the sequence of moves that make up our game

        var player: u8 = 'x';
        var board = [1]u8{'_'} ** 9;

        i += 1;

        if (i % 1000 == 0) print("We're playing game {} \n", .{i});

        for (p) |spot| {
            // each move corresponds to a board position
            board[spot] = player;
            // after each move we check if player P has won, then toggle P
            if (winning(board, player)) {
                switch (player) {
                    'x' => {
                        results.x += 1;
                    },
                    'o' => {
                        results.o += 1;
                    },
                    else => {},
                }
                break;
            }

            player = switch (player) {
                'x' => 'o',
                'o' => 'x',
                else => unreachable,
            };
        }
        results.draw += 1;

        if (i % 1000 == 0) print("Current stats: {any} \n", .{results});
    }
    print("Final stats: {any} \n", .{results});
}

// check if player won tic tac toe
pub fn winning(board: [9]u8, player: u8) bool {
    // const xxx = [3]u8{ player, player, player };
    const xxx = player;

    // rows
    {
        const b0 = std.mem.allEqual(u8, board[0..3], xxx);
        const b1 = std.mem.allEqual(u8, board[3..6], xxx);
        const b2 = std.mem.allEqual(u8, board[6..9], xxx);
        if (b0 or b1 or b2) return true;
    }

    // columns
    {
        const b0 = std.mem.allEqual(u8, &[3]u8{ board[0], board[3], board[6] }, xxx);
        const b1 = std.mem.allEqual(u8, &[3]u8{ board[1], board[4], board[7] }, xxx);
        const b2 = std.mem.allEqual(u8, &[3]u8{ board[2], board[5], board[8] }, xxx);
        if (b0 or b1 or b2) return true;
    }

    // diagonals
    {
        const b0 = std.mem.allEqual(u8, &[3]u8{ board[0], board[4], board[8] }, xxx);
        const b2 = std.mem.allEqual(u8, &[3]u8{ board[2], board[4], board[6] }, xxx);
        if (b0 or b2) return true;
    }

    return false;
}
