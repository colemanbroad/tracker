const std = @import("std");
const print = std.debug.print;
const assert = std.debug.assert;
const expect = std.testing.expect;

const im = @import("imageBase.zig");

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();
var allocator = std.testing.allocator;

const x0 =     random.float(f32)*100.0;
const y0 =     random.float(f32)*100.0;

test "bin random points onto 2D grid" {
  var x:[100]f32 = undefined;
  for (x) |*v| v.* = random.float(f32)*100.0;
  var y:[100]f32 = undefined;
  for (y) |*v| v.* = random.float(f32)*100.0;

  var grid = try im.Img2D(u8).init(10,10);

  for (x) |v,i| {
    for (y) |w,j| {
      print("x-y={d}\n", .{@intToFloat(f32,i*j)*(v-w)});
    }
  }
}


