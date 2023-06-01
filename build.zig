const std = @import("std");

const Builder = @import("std").build.Builder;
const LibExeObjStep = @import("std").build.LibExeObjStep;

pub fn build(b: *Builder) void {

    // const ztracy = @import("libs/ztracy/build.zig");
    // const enable_tracy = b.option(bool, "enable-tracy", "Enable Tracy profiler") orelse true; // EDIT HERE
    // const exe_options = b.addOptions();
    // exe_options.addOption(bool, "enable_tracy", enable_tracy);
    // const options_pkg = exe_options.getPackage("build_options");

    // const step_1 = b.step("test-delaunay", "Runs the test suite");
    const test_delaunay = b.addTest(.{ .name = "test-delaunay", .root_source_file = .{ .path = "src/delaunay.zig" } });
    b.installArtifact(test_delaunay);

    // test_delaunay.addPackage(ztracy.getPkg(b, options_pkg));

    // test_delaunay.addIncludePath("src");
    // ztracy.link(test_delaunay, enable_tracy);
    // step_1.dependOn(&test_delaunay.step);

    // const step_3 = b.step("test-grid-hash", "Runs main() in grid_hash2.zig");
    const test_gridhash = b.addTest(.{ .name = "test-grid-hash", .root_source_file = .{ .path = "src/grid_hash2.zig" } });
    b.installArtifact(test_gridhash);

    // const step_4 = b.step("tracker", "Build tracker.zig as static library for python to call.");
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

    const test_basic = b.addTest(.{ .name = "test-basic", .root_source_file = .{ .path = "src/testBasic.zig" } });
    test_basic.addAnonymousModule("trace", .{ .source_file = .{ .path = "libs/trace.zig/src/main.zig" } });
    b.installArtifact(test_basic);

    const run_basic = b.addRunArtifact(test_basic);
    b.step("run-basic", "Run the test-basic suite.").dependOn(&run_basic.step);
}

fn thisDir() []const u8 {
    return std.fs.path.dirname(@src().file) orelse ".";
}
