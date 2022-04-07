const std = @import("std");
// const warn = std.debug.warn;
const print = std.debug.print;
const expect = std.testing.expect;
const assert = std.debug.assert;

const Allocator = std.mem.Allocator;
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var allocator:Allocator = gpa.allocator();


pub fn Img2D(comptime T:type) type {
  return struct {

    const This = @This();

    img:[]T, 
    nx:u32,
    ny:u32,

    pub fn init(nx:u32 , ny:u32) !This {
      return This{
        .img = try allocator.alloc(T,nx*ny),
        .nx  = nx,
        .ny  = ny,
      };
    }

    pub fn deinit(this:This) void {
      allocator.free(this.img);
    }

  };
}

test "imageBase. new img2d" {
  const mimg = Img2D(f32){
    .img = try allocator.alloc(f32,100*100),
    .nx  = 100,
    .ny  = 100,
  };
  print("mimg {d}", .{mimg.nx});

  _ = try Img2D(f32).init(100,100);
}

pub fn Img3D(comptime T:type) type {
  return struct {
    const This = @This();
    img:[]T,
    nz:u32,    
    ny:u32,
    nx:u32,
  
    pub fn init(nx:u32 , ny:u32 , nz:u32 ) !This {
      return This{
        .img = try allocator.alloc(T,nx*ny*nz),
        .nx  = nx,
        .ny  = ny,
        .nz  = nz,
      };
    }

    pub fn deinit(this:This) void {
      allocator.free(this.img);
    }

  };
}

test "imageBase. Img3D Generic" {
  var img = try allocator.alloc(f32, 50*100*200);
  const a1 = Img3D(f32){.img=img,.nz=50,.ny=100,.nx=200};
  const a2 = try Img3D(f32).init(50,100,200); // comptime
  print("{}{}", .{a1.nx , a2.nx});
}


pub fn minmax(comptime T : type, arr : []T, ) [2]T {
  var mn : T = arr[0];
  var mx : T = arr[0];
  for (arr) |val| {
    if (val < mn) mn=val;
    if (val > mx) mx=val;
  }
  return [2]T{mn,mx};
}


// const DivByZeroNormalizationError = error {}


// Checks for mx > mn
pub fn normAffine(data : []f32, mn: f32, mx: f32) !void {
    expect(mx>mn) catch {return error.DivByZeroNormalizationError;};
    for (data) |*v| v.* = (v.* - mn) / (mx-mn);
}

// Caller guarantees mx > mn
pub fn normAffineNoErr(data : []f32, mn: f32, mx: f32) void {
    // assert(mx>mn);
    if (mn==mx) {
        print("\nWARNING (normAffineNoErr): NO Contrast. min==max.\n", .{});
        for (data) |*v| v.* = 0;
    } else {
        for (data) |*v| v.* = (v.* - mn) / (mx-mn);
    }
    }

pub fn saveF32AsTGAGreyNormed(data: []f32, h: u16, w: u16, name: []const u8,) !void {
    const rgba = try allocator.alloc(u8, 4 * data.len);
    // for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else clamp(@floatToInt(u8,data[i / 4]),0,255);
    const mnmx = minmax(f32,data);
    normAffineNoErr(data, mnmx[0], mnmx[1]);
    for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else @floatToInt(u8, data[i / 4]*255); // get negative value?
    try saveU8AsTGA(rgba, h, w, name);
}

pub fn saveF32AsTGAGreyNormedCam(cam:anytype, name: []const u8,) !void {
    const rgba = try allocator.alloc(u8, 4 * cam.screen.len);
    // for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else clamp(@floatToInt(u8,cam[i / 4]),0,255);
    const mnmx = minmax(f32,cam.screen);
    normAffineNoErr(cam.screen, mnmx[0], mnmx[1]);
    for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else @floatToInt(u8, cam.screen[i / 4]*255); // get negative value?
    try saveU8AsTGA(rgba, @intCast(u16,cam.nyPixels), @intCast(u16,cam.nxPixels), name);
}

pub fn saveF32Img2D(img:Img2D(f32), name:[]const u8) !void {
    const rgba = try allocator.alloc(u8, 4 * img.img.len);
    // for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else clamp(@floatToInt(u8,img.img[i / 4]),0,255);
    const mnmx = minmax(f32,img.img);
    normAffineNoErr(img.img, mnmx[0], mnmx[1]);
    for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else @floatToInt(u8, img.img[i / 4]*255); // get negative value?
    try saveU8AsTGA(rgba, @intCast(u16,img.ny), @intCast(u16,img.nx), name);

}

pub fn saveU8AsTGAGrey(data: []u8, h: u16, w: u16, name: []const u8) !void {
    const rgba = try allocator.alloc(u8, 4 * data.len);
    for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else data[i / 4];
    try saveU8AsTGA(rgba, h, w, name);
}

pub fn saveU8AsTGA(data: []u8, h: u16, w: u16, name: []const u8) !void {
    var cwd = std.fs.cwd();

    cwd.deleteFile(name) catch {};

    var out = try cwd.createFile(name, .{});
    defer out.close();
    errdefer cwd.deleteFile(name) catch {};
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
    var cwd = std.fs.cwd();

    cwd.deleteFile(name) catch {};

    var out = try cwd.createFile(name, .{});
    defer out.close();
    errdefer cwd.deleteFile(name) catch {};
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

    const h = @intCast(u16, pic.ny);
    const w = @intCast(u16, pic.nx);
    try writer.writeIntLittle(u16, w); //u16, @truncate(u16, self.width));
    try writer.writeIntLittle(u16, h); //u16, @truncate(u16, self.height));

    try writer.writeAll(&[_]u8{
        32, // Bit depth
        0, // Image descriptor
    });

    try writer.writeAll(std.mem.sliceAsBytes(pic.img));
}

// pub fn saveU8AsTGAGrey(allocator : *std.mem.Allocator, data: []u8, h: u16, w: u16, name: []const u8) !void {
//     const rgba = try allocator.alloc(u8, 4 * data.len);
//     for (rgba) |*v, i| v.* = if (i % 4 == 3) 255 else data[i / 4];
//     try saveU8AsTGA(rgba, h, w, name);
// }


test "imageBase. saveU8AsTGAGrey" {
    print("\n", .{});
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // var grey = .{0}**(2^10);
    var grey = std.mem.zeroes([1 << 10]u8);
    for (grey) |*v, i| v.* = @intCast(u8, i % 256);
    print("\n number ;;; {} \n", .{1 << 5});
    try saveU8AsTGAGrey(&grey, 1 << 5, 1 << 5, "testArtifacts/correct.tga");
    print("\n", .{});
}

test "imageBase. saveU8AsTGAGrey (h & w too small)" {
    print("\n", .{});
    // var grey = .{0}**(2^10);
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var grey = std.mem.zeroes([1 << 10]u8);
    for (grey) |*v, i| v.* = @intCast(u8, i % 256);
    print("\n number ;;; {} \n", .{1 << 6});
    try saveU8AsTGAGrey(&grey, 1 << 5, 1 << 4, "testArtifacts/height_width_too_small.tga");
    print("\n", .{});
}

test "imageBase. saveU8AsTGAGrey (h & w too big)" {
    print("\n", .{});
    // var grey = .{0}**(2^10);
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var grey = std.mem.zeroes([1 << 10]u8);
    for (grey) |*v, i| v.* = @intCast(u8, i % 256);
    print("\n number ;;; {} \n", .{1 << 6});
    try saveU8AsTGAGrey(&grey, 1 << 5, 1 << 6, "testArtifacts/height_width_too_big.tga");
    print("\n", .{});
}

test "imageBase. saveU8AsTGA" {
    print("\n", .{});
    var rgba = std.mem.zeroes([(1<<10) * 4]u8);
    for (rgba) |*v, i| v.* = bl: {
        const x = switch (i%4) {
                0 => i%255,     // red
                1 => (2*i)%255, // blue
                2 => (3*i)%255, // green
                3 => 255,       // alpha
                else => unreachable,
            };
        break :bl @intCast(u8,x);
        };
    // const x = @intCast(u8, i % 256); // alpha channel changes too!
    try saveU8AsTGA(&rgba, 1 << 5, 1 << 5, "testArtifacts/multicolor.tga");
    print("\n", .{});
}
