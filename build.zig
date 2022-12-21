const std = @import("std");

const Builder = @import("std").build.Builder;
const LibExeObjStep = @import("std").build.LibExeObjStep;

pub fn build(b: *Builder) void {

    const ztracy = @import("libs/ztracy/build.zig");
    const enable_tracy = b.option(bool, "enable-tracy", "Enable Tracy profiler") orelse true; // EDIT HERE
    const exe_options = b.addOptions();
    exe_options.addOption(bool, "enable_tracy", enable_tracy);
    const options_pkg = exe_options.getPackage("build_options");

    const step_1 = b.step("test-delaunay", "Runs the test suite");
    {
        const exe = b.addTest("src/delaunay.zig");
        // exe.addIncludePath("src");
        exe.addPackage(ztracy.getPkg(b, options_pkg));
        ztracy.link(exe, enable_tracy);
        step_1.dependOn(&exe.step);
    }

    const step_3 = b.step("test-grid-hash", "Runs main() in grid_hash2.zig");
    {
        const exe = b.addTest("src/grid_hash2.zig");
        step_3.dependOn(&exe.step);
    }

    const step_4 = b.step("tracker", "Build tracker.zig as static library for python to call.");
    {
        // const exe = b.addExecutable("track", "track.zig");
        // const lib = b.addSharedLibrary("track", "track.zig", .unversioned);
        const exe = b.addStaticLibrary("track", "src/tracker.zig");
        exe.install();
    }

    const runall = b.step("run","Run and install everything");
    {
        runall.dependOn(step_1);
        runall.dependOn(step_4);
        runall.dependOn(step_3);
    }
}

fn thisDir() []const u8 {
    return std.fs.path.dirname(@src().file) orelse ".";
}