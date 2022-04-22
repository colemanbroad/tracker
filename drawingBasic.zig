const std = @import("std");
const im = @import("imageBase.zig");

const Img2D = im.Img2D;

var allocator = std.testing.allocator;
const print = std.debug.print;
const assert = std.debug.assert;

const test_home = "/Users/broaddus/Desktop/work/zig-tracker/test-artifacts/drawingBasic/";
test {
    std.testing.refAllDecls(@This());
}

// Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D
// Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D
// Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D 👇 Drawing on Img2D

pub inline fn inbounds(img: anytype, px: anytype) bool {
    if (0 <= px[0] and px[0] < img.nx and 0 <= px[1] and px[1] < img.ny) return true else return false;
}

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

test "drawing. draw a simple yellow line" {
    const pic = try Img2D([4]u8).init(600, 400);
    defer pic.deinit();
    drawLine([4]u8, pic, 10, 0, 500, 100, .{ 0, 255, 255, 255 });
    try im.saveRGBA(pic, test_home ++ "testeroo.tga");
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

// just a 1px circle outline. tested with delaunay circumcircles.
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
