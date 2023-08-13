const std = @import("std");

const Builder = @import("std").Build.Builder;
const CompileStep = @import("std").Build.CompileStep;
const LibExeObjStep = @import("std").Build.LibExeObjStep;

fn addTracy(b: *Builder, exe: *CompileStep) !void {
    var default_target: std.zig.CrossTarget = .{};
    const target = b.standardTargetOptions(.{ .default_target = default_target });

    const tracy = b.option([]const u8, "tracy", "Enable Tracy integration. Supply path to Tracy source");
    const tracy_callstack = b.option(bool, "tracy-callstack", "Include callstack information with Tracy data. Does nothing if -Dtracy is not provided") orelse (tracy != null);
    const tracy_allocation = b.option(bool, "tracy-allocation", "Include allocation information with Tracy data. Does nothing if -Dtracy is not provided") orelse (tracy != null);

    const exe_options = b.addOptions();
    exe.addOptions("build_options", exe_options);

    exe_options.addOption(bool, "enable_tracy", tracy != null);
    exe_options.addOption(bool, "enable_tracy_callstack", tracy_callstack);
    exe_options.addOption(bool, "enable_tracy_allocation", tracy_allocation);

    if (tracy) |tracy_path| {
        const client_cpp = std.fs.path.join(
            b.allocator,
            &[_][]const u8{ tracy_path, "public", "TracyClient.cpp" },
        ) catch unreachable;

        // On mingw, we need to opt into windows 7+ to get some features required by tracy.
        const tracy_c_flags: []const []const u8 = if (target.isWindows() and target.getAbi() == .gnu)
            &[_][]const u8{ "-DTRACY_ENABLE=1", "-fno-sanitize=undefined", "-D_WIN32_WINNT=0x601" }
        else
            &[_][]const u8{ "-DTRACY_ENABLE=1", "-fno-sanitize=undefined" };

        exe.addIncludePath(tracy_path);
        exe.addCSourceFile(client_cpp, tracy_c_flags);

        // if (!enable_llvm) {
        exe.linkSystemLibraryName("c++");
        // }
        exe.linkLibC();

        if (target.isWindows()) {
            exe.linkSystemLibrary("dbghelp");
            exe.linkSystemLibrary("ws2_32");
        }
    }
}

pub fn addSDL(cs: *CompileStep) void {
    cs.addLibraryPath(.{ .path = "/opt/homebrew/Cellar/sdl2/2.26.3/lib/" });
    cs.addIncludePath(.{ .path = "/opt/homebrew/Cellar/sdl2/2.26.3/include/" });
    cs.linkSystemLibraryName("SDL2");
}

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
    // lib_tracker.addAnonymousModule("trace", .{ .source_file = .{ .path = "libs/trace.zig/src/main.zig" } });
    b.installArtifact(lib_tracker);

    const exe_tracker = b.addExecutable(.{
        .name = "exe-track",
        .root_source_file = .{ .path = "src/tracker.zig" },
        .target = .{},
        .optimize = .Debug,
    });
    addSDL(exe_tracker);
    b.installArtifact(exe_tracker);

    // const test_tracker = b.addTest(.{
    //     .name = "test-tracker",
    //     .root_source_file = .{ .path = "src/tracker.zig" },
    //     .filter = "generateTracking",
    // });
    // addSDL(test_tracker);
    // // test_tracker.addAnonymousModule("trace", .{ .source_file = .{ .path = "libs/trace.zig/src/main.zig" } });
    // b.installArtifact(test_tracker);

    const kdtree2d = b.addExecutable(.{
        .name = "exe-kdtree2d",
        .root_source_file = .{ .path = "src/kdtree2d.zig" },
        // .optimize = .ReleaseSafe,
        // .optimize = .ReleaseSmall,
        // .optimize = .ReleaseFast,
        .optimize = .Debug,
    });
    addSDL(kdtree2d);
    // kdtree2d.addAnonymousModule("trace", .{ .source_file = .{ .path = "libs/trace.zig/src/main.zig" } });
    // try addTracy(b, kdtree2d);
    b.installArtifact(kdtree2d);

    // const run_basic = b.addRunArtifact(test_basic);
    // b.step("run-basic", "Run the test-basic suite.").dependOn(&run_basic.step);
}

fn thisDir() []const u8 {
    return std.fs.path.dirname(@src().file) orelse ".";
}
