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
    // for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else clamp(@floatToInt(u8,data[i / 4]),0,255);
    const mnmx = minmax(f32, data);
    normAffineNoErr(data, mnmx[0], mnmx[1]);
    for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else @floatToInt(u8, data[i / 4] * 255); // get negative value?
    try saveU8AsTGA(rgba, h, w, name);
}

/// deprecated
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

/// deprecated
pub fn saveF32Img2D(img: Img2D(f32), name: []const u8) !void {
    const rgba = try allocator.alloc(u8, 4 * img.img.len);
    defer allocator.free(rgba);
    // for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else clamp(@floatToInt(u8,img.img[i / 4]),0,255);
    const mnmx = minmax(f32, img.img);
    normAffineNoErr(img.img, mnmx[0], mnmx[1]);
    for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else @floatToInt(u8, img.img[i / 4] * 255); // get negative value?
    try saveU8AsTGA(rgba, @intCast(u16, img.ny), @intCast(u16, img.nx), name);
}

/// deprecated
pub fn saveU8AsTGAGrey(data: []u8, h: u16, w: u16, name: []const u8) !void {
    const rgba = try allocator.alloc(u8, 4 * data.len);
    defer allocator.free(rgba);
    for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else data[i / 4];
    try saveU8AsTGA(rgba, h, w, name);
}

pub fn saveU8AsTGA(data: []u8, h: u16, w: u16, name: []const u8) !void {

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
        const idx = @intCast(u32, x0) + img.nx * @intCast(u32, y0);
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
            const idx = @intCast(u32, x0) + img.nx * @intCast(u32, y0);
            img.img[idx] = val;
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
    try saveRGBA(pic, "../test-artifacts/testeroo.tga");
}

pub fn drawCircle(comptime T: type, pic: Img2D(T), x0: i32, y0: i32, r: i32, val: T) void {
    var idx: i32 = 0;
    while (idx < 4 * r * r) : (idx += 1) {
        const dx = @mod(idx, 2 * r) - r;
        const dy = @divFloor(idx, 2 * r) - r;
        const x = x0 + dx;
        const y = y0 + dy;
        if (inbounds(pic, .{ x, y }) and dx * dx + dy * dy <= r * r) {
            const imgigx = @intCast(u31, x) + pic.nx * @intCast(u31, y);
            pic.img[imgigx] = val;
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
    const nx = @intCast(i32, pic.nx);
    var idx: usize = undefined;

    while (x < 0) {
        x0 = xm - x;
        y0 = ym + y;
        if (inbounds(pic, .{ x0, y0 })) {
            idx = @intCast(usize, x0 + nx * y0);
            pic.img[idx] = val;
        }
        x0 = xm + x;
        y0 = ym + y;
        if (inbounds(pic, .{ x0, y0 })) {
            idx = @intCast(usize, x0 + nx * y0);
            pic.img[idx] = val;
        }
        x0 = xm - x;
        y0 = ym - y;
        if (inbounds(pic, .{ x0, y0 })) {
            idx = @intCast(usize, x0 + nx * y0);
            pic.img[idx] = val;
        }
        x0 = xm + x;
        y0 = ym - y;
        if (inbounds(pic, .{ x0, y0 })) {
            idx = @intCast(usize, x0 + nx * y0);
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
