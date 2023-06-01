// pub fn thisDir() []const u8 {
//     return std.fs.path.dirname(@src().file) orelse ".";
// }

// no c deps. minimal old code.
test {
    _ = @import("delaunay.zig");
    _ = @import("geometry.zig");
    _ = @import("grid_hash2.zig");
    _ = @import("image_base.zig");
    _ = @import("lil1DTree.zig");
    _ = @import("mesh.zig");
    _ = @import("ok2DTree.zig");
    // _ = @import("playground.zig");
    _ = @import("render.zig");
    _ = @import("tracker.zig");
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
