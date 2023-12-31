const std = @import("std");
// const warn = std.debug.warn;
const print = std.debug.print;
const expect = std.testing.expect;
const assert = std.debug.assert;

const Allocator = std.mem.Allocator;
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// pub var allocator = std.testing.allocator;
// const root = @import("root");
// const test_artifacts = @import("root").thisDir() ++ "test-artifacts/";
const test_home = "/Users/broaddus/work/isbi/zig-tracker/test-artifacts/image_base/";

test {
    std.testing.refAllDecls(@This());
}

// 2D image with access patter idx = x*ny + y
// i.e. y is the fast-changing variable
pub fn Img2D(comptime T: type) type {
    return struct {
        const This = @This();

        img: []T,
        nx: u32,
        ny: u32,

        pub fn get(this: This, x: usize, y: usize) T {
            return this.img[x * this.ny + y];
        }

        pub fn set(this: This, x: usize, y: usize, v: T) void {
            this.img[x * this.ny + y] = v;
        }

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

test "test imageBase. new img2d" {
    const mimg = Img2D(f32){
        .img = try allocator.alloc(f32, 100 * 100),
        .nx = 100,
        .ny = 100,
    };
    defer mimg.deinit();
    // print("mimg {d}", .{mimg.nx});

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

test "test imageBase. Img3D Generic" {
    var img = try allocator.alloc(f32, 50 * 100 * 200);
    const a1 = Img3D(f32){ .img = img, .nz = 50, .ny = 100, .nx = 200 };
    defer a1.deinit();
    const a2 = try Img3D(f32).init(50, 100, 200); // comptime
    defer a2.deinit();
    // print("{}{}", .{ a1.nx, a2.nx });
}

pub inline fn inbounds(img: anytype, px: anytype) bool {
    if (0 <= px[0] and px[0] < img.nx and 0 <= px[1] and px[1] < img.ny) return true else return false;
}

// pub fn minmaxArray()

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

/// Checks for mx > mn
pub fn normAffine(data: []f32, mn: f32, mx: f32) !void {
    expect(mx > mn) catch {
        return error.DivByZeroNormalizationError;
    };
    for (data) |*v| v.* = (v.* - mn) / (mx - mn);
}

/// Caller guarantees mx > mn
pub fn normAffineNoErr(data: []f32, mn: f32, mx: f32) void {
    // assert(mx>mn);
    if (mn == mx) {
        print("\nWARNING (normAffineNoErr): NO Contrast. min==max.\n", .{});
        for (data) |*v| v.* = 0;
    } else {
        for (data) |*v| v.* = (v.* - mn) / (mx - mn);
    }
}

/// deprecated
pub fn saveF32AsTGAGreyNormed(
    data: []f32,
    h: u16,
    w: u16,
    name: []const u8,
) !void {
    const rgba = try allocator.alloc(u8, 4 * data.len);
    defer allocator.free(rgba);
    // for (&rgba) |*v, i| v.* = if (i % 4 == 3) 255 else clamp(@floatToInt(u8,data[i / 4]),0,255);
    const mnmx = minmax(f32, data);
    normAffineNoErr(data, mnmx[0], mnmx[1]);
    for (rgba, 0..) |*v, i| v.* = if (i % 4 == 3) 255 else @as(u8, @intFromFloat(data[i / 4] * 255)); // get negative value?
    try saveU8AsTGA(rgba, h, w, name);
}

/// deprecated
pub fn saveF32AsTGAGreyNormedCam(
    cam: anytype,
    name: []const u8,
) !void {
    var rgba = try allocator.alloc(u8, 4 * cam.screen.len);
    defer allocator.free(rgba);
    // for (&rgba) |*v, i| v.* = if (i % 4 == 3) 255 else clamp(@floatToInt(u8,cam[i / 4]),0,255);
    const mnmx = minmax(f32, cam.screen);
    normAffineNoErr(cam.screen, mnmx[0], mnmx[1]);
    for (rgba, 0..) |*v, i| v.* = if (i % 4 == 3) 255 else @as(u8, @intFromFloat(cam.screen[i / 4] * 255)); // get negative value?
    try saveU8AsTGA(rgba, @as(u16, @intCast(cam.nyPixels)), @as(u16, @intCast(cam.nxPixels)), name);
}

/// deprecated
pub fn saveF32Img2D(img: Img2D(f32), name: []const u8) !void {
    var rgba = try allocator.alloc(u8, 4 * img.img.len);
    defer allocator.free(rgba);
    // for (&rgba) |*v, i| v.* = if (i % 4 == 3) 255 else clamp(@floatToInt(u8,img.img[i / 4]),0,255);
    const mnmx = minmax(f32, img.img);
    normAffineNoErr(img.img, mnmx[0], mnmx[1]);
    for (rgba, 0..) |*v, i| v.* = if (i % 4 == 3) 255 else @as(u8, @intFromFloat(img.img[i / 4] * 255)); // get negative value?
    try saveU8AsTGA(rgba, @as(u16, @intCast(img.ny)), @as(u16, @intCast(img.nx)), name);
}

/// deprecated
pub fn saveU8AsTGAGrey(data: []u8, h: u16, w: u16, name: []const u8) !void {
    var rgba = try allocator.alloc(u8, 4 * data.len);
    defer allocator.free(rgba);
    for (rgba, 0..) |*v, i| v.* = if (i % 4 == 3) 255 else data[i / 4];
    try saveU8AsTGA(rgba, h, w, name);
}

pub fn saveU8AsTGA(data: []u8, h: u16, w: u16, name: []const u8) !void {

    // // determine if absolute path or relative path. ensure there is a filename with ".tga"
    // //
    // // remove file if already exists
    // // make path if it doesn't exist

    // const cwd = std.fs.cwd();
    const resolved = try std.fs.path.resolve(allocator, &.{name});
    defer allocator.free(resolved);
    // const dirname = std.fs.path.dirname(resolved);

    // // const basename = std.fs.path.basename(resolved);
    // // print("resolved : {s} \n" , .{resolved});
    // // print("dirname : {s} \n" , .{dirname});
    // // print("basename : {s} \n" , .{basename});

    // cwd.makePath(dirname.?) catch {};
    // // cwd.createFile(sub_path: []const u8, flags: File.CreateFlags)

    // // try std.fs.makeDirAbsolute(dirname.?);
    // // const dirnameDir = try std.fs.openDirAbsolute(dirname.?, .{});
    // // try dirnameDir.makePath("");

    // // WARNING fails when `resolved` is an existing directory...
    // // std.fs.deleteDirAbsolute(resolved) catch {};
    // std.fs.deleteFileAbsolute(resolved) catch {};
    // var out = std.fs.createFileAbsolute(resolved, .{ .exclusive = true }) catch unreachable;
    // defer out.close();
    // // errdefer cwd.deleteFile(name) catch {};

    // var outfile = try std.fs.cwd().createFile(name, .{});
    // var outfile = try std.fs.createFileAbsolute(absolute_path: []const u8, flags: File.CreateFlags)
    errdefer print("We've errored. Boo Hoo. The resolved = {s} \n\n", .{resolved});

    var outfile = try std.fs.createFileAbsolute(resolved, .{});
    defer outfile.close();
    var writer = outfile.writer();

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
    const h = @as(u16, @intCast(pic.ny));
    const w = @as(u16, @intCast(pic.nx));
    try saveU8AsTGA(data, h, w, name);
}

// Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D
// Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D
// Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D

pub fn myabs(a: anytype) @TypeOf(a) {
    if (a < 0) return -a else return a;
}

pub fn drawLine(comptime T: type, img: Img2D(T), _x0: u31, _y0: u31, x1: u31, y1: u31, val: T) void {
    var x0: i32 = _x0;
    var y0: i32 = _y0;
    const dx = myabs(x1 - x0);
    const sx: i8 = if (x0 < x1) 1 else -1;
    const dy = -myabs(y1 - y0);
    const sy: i8 = if (y0 < y1) 1 else -1;
    var err: i32 = dx + dy; //
    var e2: i32 = 0;

    while (true) {
        const idx = @as(u32, @intCast(x0)) + img.nx * @as(u32, @intCast(y0));
        img.img[idx] = val;
        e2 = 2 * err;
        if (e2 >= dy) {
            if (x0 == x1) break;
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            if (y0 == y1) break;
            err += dx;
            y0 += sy;
        }
    }
}

pub fn drawLineInBounds(comptime T: type, img: Img2D(T), _x0: i32, _y0: i32, x1: i32, y1: i32, val: T) void {
    drawLineInBounds2([4]u8, img, _x0, _y0, x1, y1, val);
    drawLineInBounds2([4]u8, img, _x0 + 1, _y0, x1 + 1, y1, val);
    drawLineInBounds2([4]u8, img, _x0, _y0 + 1, x1, y1 + 1, val);
}

fn blend(v1: [4]u8, v2: [4]u8) [4]u8 {
    return .{
        v1[0] / 2 + v2[0] / 2,
        v1[1] / 2 + v2[1] / 2,
        v1[2] / 2 + v2[2] / 2,
        v1[3] / 2 + v2[3] / 2,
    };
}
pub fn drawLineInBounds2(comptime T: type, img: Img2D(T), _x0: i32, _y0: i32, x1: i32, y1: i32, val: T) void {
    var x0 = _x0;
    var y0 = _y0;
    const dx = myabs(x1 - x0);
    const sx: i8 = if (x0 < x1) 1 else -1;
    const dy = -myabs(y1 - y0);
    const sy: i8 = if (y0 < y1) 1 else -1;
    var err: i32 = dx + dy; //
    var e2: i32 = 0;

    while (true) {
        if (inbounds(img, .{ x0, y0 })) {
            const idx = @as(u32, @intCast(x0)) + img.nx * @as(u32, @intCast(y0));
            img.img[idx] = val; // blend(img.img[idx], val);
        }
        e2 = 2 * err;
        if (e2 >= dy) {
            if (x0 == x1) break;
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            if (y0 == y1) break;
            err += dx;
            y0 += sy;
        }
    }
}

test "test imageBase. draw a simple yellow line" {
    const pic = try Img2D([4]u8).init(600, 400);
    defer pic.deinit();
    drawLine([4]u8, pic, 10, 0, 500, 100, .{ 0, 255, 255, 255 });
    try saveRGBA(pic, test_home ++ "testeroo.tga");
}

pub fn drawCircle(comptime T: type, pic: Img2D(T), x0: i32, y0: i32, _r: i32, val: T) void {
    const r = 2 * _r;
    var idx: i32 = 0;
    while (idx < 4 * r * r) : (idx += 1) {
        const dx = @mod(idx, 2 * r) - r;
        const dy = @divFloor(idx, 2 * r) - r;
        const x = x0 + dx;
        const y = y0 + dy;
        if (inbounds(pic, .{ x, y }) and dx * dx + dy * dy <= r * r) {
            const imgigx = @as(u31, @intCast(x)) + pic.nx * @as(u31, @intCast(y));
            // pic.img[imgigx] = blend(pic.img[imgigx], val);
            pic.img[imgigx] = val; //blend(pic.img[imgigx], val);
        }
    }
}

/// just a 1px circle outline. tested with delaunay circumcircles.
pub fn drawCircleOutline(pic: Img2D([4]u8), xm: i32, ym: i32, _r: i32, val: [4]u8) void {
    var r = _r;
    var x = -r;
    var y: i32 = 0;
    var err: i32 = 2 - 2 * r; // /* bottom left to top right */
    var x0: i32 = undefined;
    var y0: i32 = undefined;
    const nx = @as(i32, @intCast(pic.nx));
    var idx: usize = undefined;

    while (x < 0) {
        x0 = xm - x;
        y0 = ym + y;
        if (inbounds(pic, .{ x0, y0 })) {
            idx = @as(usize, @intCast(x0 + nx * y0));
            pic.img[idx] = val;
        }
        x0 = xm + x;
        y0 = ym + y;
        if (inbounds(pic, .{ x0, y0 })) {
            idx = @as(usize, @intCast(x0 + nx * y0));
            pic.img[idx] = val;
        }
        x0 = xm - x;
        y0 = ym - y;
        if (inbounds(pic, .{ x0, y0 })) {
            idx = @as(usize, @intCast(x0 + nx * y0));
            pic.img[idx] = val;
        }
        x0 = xm + x;
        y0 = ym - y;
        if (inbounds(pic, .{ x0, y0 })) {
            idx = @as(usize, @intCast(x0 + nx * y0));
            pic.img[idx] = val;
        }

        // setPixel(xm-x, ym+y); //                           /*   I. Quadrant +x +y */
        // setPixel(xm-y, ym-x); //                           /*  II. Quadrant -x +y */
        // setPixel(xm+x, ym-y); //                           /* III. Quadrant -x -y */
        // setPixel(xm+y, ym+x); //                           /*  IV. Quadrant +x -y */
        r = err;
        if (r <= y) {
            y += 1;
            err += y * 2 + 1;
        } //  /* e_xy+e_y < 0 */
        if (r > x or err > y) {
            x += 1;
            err += x * 2 + 1; //  /* -> x-step now */
        } //  /* e_xy+e_x > 0 or no 2nd y-step */
    }
}

test "imageBase. saveU8AsTGAGrey" {
    print("\n", .{});
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // var grey = .{0}**(2^10);
    var grey = std.mem.zeroes([1 << 10]u8);
    for (&grey, 0..) |*v, i| v.* = @as(u8, @intCast(i % 256));
    print("\n number ;;; {} \n", .{1 << 5});
    try saveU8AsTGAGrey(&grey, 1 << 5, 1 << 5, test_home ++ "correct.tga");
    print("\n", .{});
}

test "imageBase. saveU8AsTGAGrey (h & w too small)" {
    print("\n", .{});
    // var grey = .{0}**(2^10);
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var grey = std.mem.zeroes([1 << 10]u8);
    for (&grey, 0..) |*v, i| v.* = @as(u8, @intCast(i % 256));
    print("\n number ;;; {} \n", .{1 << 6});
    try saveU8AsTGAGrey(&grey, 1 << 5, 1 << 4, test_home ++ "height_width_too_small.tga");
    print("\n", .{});
}

test "imageBase. saveU8AsTGAGrey (h & w too big)" {
    print("\n", .{});
    // var grey = .{0}**(2^10);
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var grey = std.mem.zeroes([1 << 10]u8);
    for (&grey, 0..) |*v, i| v.* = @as(u8, @intCast(i % 256));
    print("\n number ;;; {} \n", .{1 << 6});
    try saveU8AsTGAGrey(&grey, 1 << 5, 1 << 6, test_home ++ "height_width_too_big.tga");
    print("\n", .{});
}

test "imageBase. saveU8AsTGA" {
    print("\n", .{});
    var rgba = std.mem.zeroes([(1 << 10) * 4]u8);
    for (&rgba, 0..) |*v, i| v.* = bl: {
        const x = switch (i % 4) {
            0 => i % 255, // red
            1 => (2 * i) % 255, // blue
            2 => (3 * i) % 255, // green
            3 => 255, // alpha
            else => unreachable,
        };
        break :bl @as(u8, @intCast(x));
    };
    // const x = @intCast(u8, i % 256); // alpha channel changes too!
    try saveU8AsTGA(&rgba, 1 << 5, 1 << 5, test_home ++ "multicolor.tga");
    print("\n", .{});
}
