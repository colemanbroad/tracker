const std = @import("std");
pub const cwd = std.fs.cwd();

test {

  const testdir = try cwd.openDir("test-artifacts/",.{});
  try testdir.setAsCwd();
  // std.testing.refAllDecls(@This());

  // _ = @import("arrayset.zig");
  // _ = @import("build.zig");
  // _ = @import("c.zig");
  // _ = @import("delaunay.zig");
  // _ = @import("drawing.zig");
  // _ = @import("geometry.zig");
  // _ = @import("imageBase.zig");
  // _ = @import("imageToys.zig");
  // _ = @import("mesh.zig");
  _ = @import("spatial.zig");
  // _ = @import("track.zig");
  // _ = @import("tracylib.zig");

}


const Str = []const u8;
pub fn mkdirIgnoreExists(dirname:Str) !void {
  cwd.makeDir(dirname) catch |e| switch (e) {
    error.PathAlreadyExists => {},
    else => return e ,
  };
}

pub fn dir_mkdirIgnoreExists(dir:std.fs.Dir, dirname:Str) !void {
  dir.makeDir(dirname) catch |e| switch (e) {
    error.PathAlreadyExists => {},
    else => return e ,
  };
}
