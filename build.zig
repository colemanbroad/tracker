const std = @import("std");

const Builder = @import("std").build.Builder;

pub fn build(b: *Builder) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();

    // const exe = b.addExecutable("track", "track.zig");
    const exe = b.addExecutable("imageToys", "imageToys.zig");
    exe.addIncludePath("tracy/");
    exe.addIncludePath(".");

    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.install();

    const tracy = b.addExecutable("tracy", "tracy/TracyClient.cpp");
    tracy.addIncludePath("tracy/");

    tracy.setTarget(target);
    tracy.setBuildMode(mode);
    tracy.install();



    // const lib = b.addStaticLibrary("track", "track.zig");
    const lib = b.addSharedLibrary("track", "track.zig", .unversioned);
    lib.setBuildMode(mode);
    lib.install();

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}

fn linkOpenCL(b: *Builder, exe : *LibExeObjStep) void {
    const mode = b.standardReleaseOptions();
    exe.setBuildMode(mode);
    exe.addIncludeDir("opencl-headers");
    // exe.addIncludeDir("./src/");
    exe.linkSystemLibrary("c");

    if (std.builtin.os.tag == .windows) {
        std.debug.warn("Windows detected, adding default CUDA SDK x64 lib search path. Change this in build.zig if needed...");
        exe.addLibPath("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64");
    } else if (std.builtin.os.tag == .macos) {
        exe.linkFramework("OpenCL");
    } else {
        exe.linkSystemLibrary("OpenCL");
    }
    
    if (exe.kind != .Test) exe.install();
}