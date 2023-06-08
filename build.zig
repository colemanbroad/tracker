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

    const kdtree2d = b.addExecutable(.{
        .name = "exe-kdtree2d",
        .root_source_file = .{ .path = "src/kdtree2d.zig" },
        // .optimize = .ReleaseSafe,
        // .optimize = .ReleaseSmall,
        .optimize = .ReleaseFast,
        // .optimize = .Debug,
    });
    kdtree2d.addLibraryPath("/opt/homebrew/Cellar/sdl2/2.26.3/lib/");
    kdtree2d.addIncludePath("/opt/homebrew/Cellar/sdl2/2.26.3/include/");
    kdtree2d.linkSystemLibraryName("SDL2");
    kdtree2d.addAnonymousModule("trace", .{ .source_file = .{ .path = "libs/trace.zig/src/main.zig" } });
    // try addTracy(b, kdtree2d);
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
