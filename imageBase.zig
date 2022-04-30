const std = @import("std");
// const warn = std.debug.warn;
const print = std.debug.print;
const expect = std.testing.expect;
const assert = std.debug.assert;

const Allocator = std.mem.Allocator;
// var gpa = std.heap.GeneralPurposeAllocator(.{}){};
// var allocator = gpa.allocator();
var allocator = std.testing.allocator;

test {
    std.testing.refAllDecls(@This());
}

pub fn Img2D(comptime T: type) type {
    return struct {
        const This = @This();

        img: []T,
        nx: u32,
        ny: u32,

        pub fn init(nx: u32, ny: u32) !This {
            return This{
                .img = try allocator.alloc(T, nx * ny),
                .nx = nx,
                .ny = ny,
            };
        }

        pub fn deinit(this: This) void {
            allocator.free(this.img);
        }
    };
}

test "imageBase. new img2d" {
    const mimg = Img2D(f32){
        .img = try allocator.alloc(f32, 100 * 100),
        .nx = 100,
        .ny = 100,
    };
    defer mimg.deinit();
    print("mimg {d}", .{mimg.nx});

    const bimg = try Img2D(f32).init(100, 100);
    defer bimg.deinit();
}

pub fn Img3D(comptime T: type) type {
    return struct {
        const This = @This();
        img: []T,
        nz: u32,
        ny: u32,
        nx: u32,

        pub fn init(nx: u32, ny: u32, nz: u32) !This {
            return This{
                .img = try allocator.alloc(T, nx * ny * nz),
                .nx = nx,
                .ny = ny,
                .nz = nz,
            };
        }

        pub fn deinit(this: This) void {
            allocator.free(this.img);
        }
    };
}

test "imageBase. Img3D Generic" {
    var img = try allocator.alloc(f32, 50 * 100 * 200);
    const a1 = Img3D(f32){ .img = img, .nz = 50, .ny = 100, .nx = 200 };
    defer a1.deinit();
    const a2 = try Img3D(f32).init(50, 100, 200); // comptime
    defer a2.deinit();
    print("{}{}", .{ a1.nx, a2.nx });
}

pub inline fn inbounds(img: anytype, px: anytype) bool {
    if (0 <= px[0] and px[0] < img.nx and 0 <= px[1] and px[1] < img.ny) return true else return false;
}

pub fn minmax(
    comptime T: type,
    arr: []T,
) [2]T {
    var mn: T = arr[0];
    var mx: T = arr[0];
    for (arr) |val| {
        if (val < mn) mn = val;
        if (val > mx) mx = val;
    }
    return [2]T{ mn, mx };
}

// const DivByZeroNormalizationError = error {}

// Checks for mx > mn
pub fn normAffine(data: []f32, mn: f32, mx: f32) !void {
    expect(mx > mn) catch {
        return error.DivByZeroNormalizationError;
    };
    for (data) |*v| v.* = (v.* - mn) / (mx - mn);
}

// Caller guarantees mx > mn
pub fn normAffineNoErr(data: []f32, mn: f32, mx: f32) void {
    // assert(mx>mn);
    if (mn == mx) {
        print("\nWARNING (normAffineNoErr): NO Contrast. min==max.\n", .{});
        for (data) |*v| v.* = 0;
    } else {
        for (data) |*v| v.* = (v.* - mn) / (mx - mn);
    }
}

pub fn saveF32AsTGAGreyNormed(
    data: []f32,
    h: u16,
    w: u16,
    name: []const u8,
) !void {
    const rgba = try allocator.alloc(u8, 4 * data.len);
    defer allocator.free(rgba);
    // for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else clamp(@floatToInt(u8,data[i / 4]),0,255);
    const mnmx = minmax(f32, data);
    normAffineNoErr(data, mnmx[0], mnmx[1]);
    for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else @floatToInt(u8, data[i / 4] * 255); // get negative value?
    try saveU8AsTGA(rgba, h, w, name);
}

pub fn saveF32AsTGAGreyNormedCam(
    cam: anytype,
    name: []const u8,
) !void {
    const rgba = try allocator.alloc(u8, 4 * cam.screen.len);
    defer allocator.free(rgba);
    // for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else clamp(@floatToInt(u8,cam[i / 4]),0,255);
    const mnmx = minmax(f32, cam.screen);
    normAffineNoErr(cam.screen, mnmx[0], mnmx[1]);
    for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else @floatToInt(u8, cam.screen[i / 4] * 255); // get negative value?
    try saveU8AsTGA(rgba, @intCast(u16, cam.nyPixels), @intCast(u16, cam.nxPixels), name);
}

pub fn saveF32Img2D(img: Img2D(f32), name: []const u8) !void {
    const rgba = try allocator.alloc(u8, 4 * img.img.len);
    defer allocator.free(rgba);
    // for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else clamp(@floatToInt(u8,img.img[i / 4]),0,255);
    const mnmx = minmax(f32, img.img);
    normAffineNoErr(img.img, mnmx[0], mnmx[1]);
    for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else @floatToInt(u8, img.img[i / 4] * 255); // get negative value?
    try saveU8AsTGA(rgba, @intCast(u16, img.ny), @intCast(u16, img.nx), name);
}

pub fn saveU8AsTGAGrey(data: []u8, h: u16, w: u16, name: []const u8) !void {
    const rgba = try allocator.alloc(u8, 4 * data.len);
    defer allocator.free(rgba);
    for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else data[i / 4];
    try saveU8AsTGA(rgba, h, w, name);
}

pub fn saveU8AsTGA(data: []u8, h: u16, w: u16, name: []const u8) !void {

    // var filename =
    // determine if absolute path or relative path. ensure there is a filename with ".tga"
    // remove file if already exists
    // make path if it doesn't exist

    const cwd = std.fs.cwd();
    const resolved = try std.fs.path.resolve(allocator, &.{name});
    defer allocator.free(resolved);
    const dirname = std.fs.path.dirname(resolved);
    // const basename = std.fs.path.basename(resolved);

    // print("resolved : {s} \n" , .{resolved});
    // print("dirname : {s} \n" , .{dirname});
    // print("basename : {s} \n" , .{basename});

    cwd.makePath(dirname.?) catch {};
    // cwd.createFile(sub_path: []const u8, flags: File.CreateFlags)

    // try std.fs.makeDirAbsolute(dirname.?);
    // const dirnameDir = try std.fs.openDirAbsolute(dirname.?, .{});
    // try dirnameDir.makePath("");

    // WARNING fails when `resolved` is an existing directory...
    // std.fs.deleteDirAbsolute(resolved) catch {};
    std.fs.deleteFileAbsolute(resolved) catch {};
    var out = std.fs.createFileAbsolute(resolved, .{ .exclusive = true }) catch unreachable;

    // print("dirname and basename \n {s} \n {s} \n" , .{dirname,basename});

    // std.fs.path.isAbsolute(name);
    // std.fs.makeDirAbsolute(name2) catch {};

    // var cwd = std.fs.cwd();
    // var out = try cwd.createFile(resolved, .{});
    defer out.close();
    // errdefer cwd.deleteFile(name) catch {};

    var writer = out.writer();

    try writer.writeAll(&[_]u8{
        0, // ID length
        0, // No color map
        2, // Unmapped RGB
        0,
        0,
        0,
        0,
        0, // No color map
        0,
        0, // X origin
        0,
        0, // Y origin
    });

    // try writer.writeIntLittle(u16, @truncate(u16, self.width));
    // try writer.writeIntLittle(u16, @truncate(u16, self.height));

    try writer.writeIntLittle(u16, w); //u16, @truncate(u16, self.width));
    try writer.writeIntLittle(u16, h); //u16, @truncate(u16, self.height));

    try writer.writeAll(&[_]u8{
        32, // Bit depth
        0, // Image descriptor
    });

    try writer.writeAll(data);
}

pub fn saveRGBA(pic: Img2D([4]u8), name: []const u8) !void {
    const data = std.mem.sliceAsBytes(pic.img);
    const h = @intCast(u16, pic.ny);
    const w = @intCast(u16, pic.nx);
    try saveU8AsTGA(data, h, w, name);
}
