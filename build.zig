const std = @import("std");

const Builder = @import("std").build.Builder;
const LibExeObjStep = @import("std").build.LibExeObjStep;

pub fn build(b: *Builder) void {
    const test_delaunay = b.addTest(.{ .name = "test-delaunay", .root_source_file = .{ .path = "src/delaunay.zig" } });
    b.installArtifact(test_delaunay);

    const test_gridhash = b.addTest(.{ .name = "test-grid-hash", .root_source_file = .{ .path = "src/grid_hash2.zig" } });
    b.installArtifact(test_gridhash);

    const lib_tracker = b.addSharedLibrary(.{
        .name = "track",
        .root_source_file = .{ .path = "src/tracker.zig" },
        .target = .{},
        .optimize = .Debug,
    });
    lib_tracker.addAnonymousModule("trace", .{ .source_file = .{ .path = "libs/trace.zig/src/main.zig" } });
    b.installArtifact(lib_tracker);

    const test_tracker = b.addTest(.{ .name = "test-tracker", .root_source_file = .{ .path = "src/tracker.zig" } });
    test_tracker.addAnonymousModule("trace", .{ .source_file = .{ .path = "libs/trace.zig/src/main.zig" } });
    b.installArtifact(test_tracker);

    const test_tree2d = b.addExecutable(.{
        .name = "test-tree2d",
        .root_source_file = .{ .path = "src/ok2DTree.zig" },
        .optimize = .Debug,
    });
    test_tree2d.addLibraryPath("/opt/homebrew/Cellar/sdl2/2.26.3/lib/");
    test_tree2d.addIncludePath("/opt/homebrew/Cellar/sdl2/2.26.3/include/");
    test_tree2d.linkSystemLibraryName("SDL2");
    b.installArtifact(test_tree2d);

    const kdtree2d = b.addExecutable(.{ .name = "exe-kdtree2d", .root_source_file = .{ .path = "src/kdtree2d.zig" } });
    kdtree2d.addLibraryPath("/opt/homebrew/Cellar/sdl2/2.26.3/lib/");
    kdtree2d.addIncludePath("/opt/homebrew/Cellar/sdl2/2.26.3/include/");
    kdtree2d.linkSystemLibraryName("SDL2");
    kdtree2d.addAnonymousModule("trace", .{ .source_file = .{ .path = "libs/trace.zig/src/main.zig" } });
    b.installArtifact(kdtree2d);

    const sdl_sdltest = b.addExecutable(.{ .name = "sdl-sdltest", .root_source_file = .{ .path = "src/sdltest.c" } });
    sdl_sdltest.addLibraryPath("/opt/homebrew/Cellar/sdl2/2.26.3/lib/");
    sdl_sdltest.addIncludePath("/opt/homebrew/Cellar/sdl2/2.26.3/include/");
    sdl_sdltest.linkSystemLibraryName("SDL2");
    b.installArtifact(sdl_sdltest);

    const sdl_sdltest2 = b.addExecutable(.{ .name = "sdl-01_hello_SDL", .root_source_file = .{ .path = "src/01_hello_SDL.cpp" } });
    sdl_sdltest2.addLibraryPath("/opt/homebrew/Cellar/sdl2/2.26.3/lib/");
    sdl_sdltest2.addIncludePath("/opt/homebrew/Cellar/sdl2/2.26.3/include/");
    sdl_sdltest2.linkSystemLibraryName("SDL2");
    b.installArtifact(sdl_sdltest2);

    // const test_basic = b.addTest(.{ .name = "test-basic", .root_source_file = .{ .path = "src/testBasic.zig" } });
    // test_basic.addAnonymousModule("trace", .{ .source_file = .{ .path = "libs/trace.zig/src/main.zig" } });
    // b.installArtifact(test_basic);

    // const run_basic = b.addRunArtifact(test_basic);
    // b.step("run-basic", "Run the test-basic suite.").dependOn(&run_basic.step);
}

fn thisDir() []const u8 {
    return std.fs.path.dirname(@src().file) orelse ".";
}
