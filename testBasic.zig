pub fn thisDir() []const u8 {
    return std.fs.path.dirname(@src().file) orelse ".";
}

// no c deps. minimal old code.
test {
    _ = @import("cam3d.zig");
    _ = @import("delaunay.zig");
    _ = @import("imageBase.zig");
    _ = @import("drawing_basic.zig");
    _ = @import("geometry.zig");
    _ = @import("track.zig");
    _ = @import("spatial.zig");
}

// test {
//   // _ = @import("arrayset.zig");
//   // _ = @import("build.zig");
// }

// test {
//   _ = @import("c.zig");
//   _ = @import("drawing.zig");
//   _ = @import("delaunay_image.zig");
//   _ = @import("imageToys.zig");
// }
