pub const cc = struct {
    pub usingnamespace @cImport({
        @cInclude("SDL2/SDL.h");
    });
};

/// For some reason, this isn't parsed automatically. According to SDL docs, the
/// surface pointer returned is optional!
extern fn SDL_GetWindowSurface(window: *cc.SDL_Window) ?*cc.SDL_Surface;

const im = @import("image_base.zig");
const std = @import("std");

const milliTimestamp = std.time.milliTimestamp;

const SDL_WINDOWPOS_UNDEFINED = @bitCast(c_int, cc.SDL_WINDOWPOS_UNDEFINED_MASK);

pub fn initSDL() !void {
    if (cc.SDL_Init(cc.SDL_INIT_VIDEO) != 0) return error.SDLInitializationFailed;
}

pub fn quitSDL() void {
    cc.SDL_Quit();
}

pub const Window = struct {
    sdl_window: *cc.SDL_Window,
    surface: *cc.SDL_Surface,
    pix: im.Img2D([4]u8),
    sdl_event: cc.SDL_Event,

    must_quit: bool = false,
    needs_update: bool,
    update_count: u64,
    windowID: u32,
    nx: u32,
    ny: u32,

    const This = @This();

    /// WARNING: c managed heap memory mixed with our custom allocator
    pub fn init(nx: u32, ny: u32) !This {
        // var t1: i64 = undefined;
        // var t2: i64 = undefined;

        // window = SDL_CreateWindow( "SDL Tutorial", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN );

        // t1 = milliTimestamp();
        const window = cc.SDL_CreateWindow(
            "SDL Window",
            SDL_WINDOWPOS_UNDEFINED,
            SDL_WINDOWPOS_UNDEFINED,
            @intCast(c_int, nx),
            @intCast(c_int, ny),
            // cc.SDL_WINDOW_OPENGL,
            cc.SDL_WINDOW_SHOWN,
        ) orelse {
            cc.SDL_Log("Unable to create window: %s", cc.SDL_GetError());
            return error.SDLInitializationFailed;
        };
        // t2 = milliTimestamp();
        // print("CreateWindow [{}ms]\n", .{t2 - t1});

        // t1 = milliTimestamp();
        const surface = SDL_GetWindowSurface(window) orelse {
            cc.SDL_Log("Unable to get window surface: %s", cc.SDL_GetError());
            return error.SDLInitializationFailed;
        };
        // t2 = milliTimestamp();
        // print("SDL_GetWindowSurface [{}ms]\n", .{t2 - t1});

        var pix: [][4]u8 = undefined;
        pix.ptr = @ptrCast([*][4]u8, surface.pixels.?);
        pix.len = nx * ny;
        var img = im.Img2D([4]u8){
            .img = pix,
            .nx = nx,
            .ny = ny,
        };

        const res = .{
            .sdl_window = window,
            .surface = surface,
            .pix = img,
            .needs_update = false,
            .update_count = 0,
            .windowID = cc.SDL_GetWindowID(window),
            .sdl_event = undefined,
            .nx = nx,
            .ny = ny,
        };

        res.pix.img[3 * nx + 50] = .{ 255, 255, 255, 255 };
        res.pix.img[3 * nx + 51] = .{ 255, 255, 255, 255 };
        res.pix.img[3 * nx + 52] = .{ 255, 255, 255, 255 };
        res.pix.img[3 * nx + 53] = .{ 255, 255, 255, 255 };

        return res;
    }

    pub fn update(this: This) !void {
        // _ = cc.SDL_LockSurface(this.surface);
        // for (this.pix.img, 0..) |v, i| {
        //     this.pix.img[i] = v;
        // }

        // for (this.pix.img, 0..) |p, i| {
        //     this.surface.pixels[i] = p;
        // }

        // this.surface.pixels = &this.pix.img[0];
        // this.setPixel(x: c_int, y: c_int, pixel: [4]u8)
        // cc.SDL_UnlockSurface(this.surface);

        const err = cc.SDL_UpdateWindowSurface(this.sdl_window);
        if (err != 0) {
            cc.SDL_Log("Error updating window surface: %s", cc.SDL_GetError());
            return error.SDLUpdateWindowFailed;
        }
    }

    pub fn setPixel(this: *This, x: c_int, y: c_int, pixel: [4]u8) void {
        const target_pixel = @ptrToInt(this.surface.pixels) +
            @intCast(usize, y) * @intCast(usize, this.surface.pitch) +
            @intCast(usize, x) * 4;
        @intToPtr(*u32, target_pixel).* = @bitCast(u32, pixel);
    }

    pub fn setPixels(this: *This, buffer: [][4]u8) void {
        _ = cc.SDL_LockSurface(this.surface);
        for (buffer, 0..) |v, i| {
            this.pix.img[i] = v;
        }
        cc.SDL_UnlockSurface(this.surface);
    }

    pub fn markBounds(this: *This) void {
        // _ = cc.SDL_LockSurface(this.surface);
        // TOP LEFT BLUE
        im.drawCircle([4]u8, this.pix, 0, 0, 13, .{ 255, 0, 0, 255 });
        // TOP RIGHT GREEN
        im.drawCircle([4]u8, this.pix, @intCast(i32, this.nx), 0, 13, .{ 0, 255, 0, 255 });
        // BOT LEFT RED
        im.drawCircle([4]u8, this.pix, 0, @intCast(i32, this.ny), 13, .{ 0, 0, 255, 255 });
        // BOT RIGHT WHITE
        im.drawCircle([4]u8, this.pix, @intCast(i32, this.nx), @intCast(i32, this.ny), 13, .{ 255, 255, 255, 255 });
        // cc.SDL_UnlockSurface(this.surface);
    }

    // An event loop that only moves forward on KeyDown.
    // Use this in combination with drawing to the window to visually step
    // through algorithms! This is the standard basic event loop, but should
    // be overridden for more interactivity.
    pub fn awaitKeyPressAndUpdateWindow(self: *This) void {
        if (self.must_quit) return;
        while (true) {
            _ = cc.SDL_WaitEvent(&self.sdl_event);
            if (self.sdl_event.type == cc.SDL_KEYDOWN) {
                switch (self.sdl_event.key.keysym.sym) {
                    cc.SDLK_q => std.os.exit(0),
                    cc.SDLK_f => {
                        self.must_quit = true;
                    }, // finish
                    else => {
                        self.update() catch {};
                        break;
                    },
                }
            }
        }
    }

    // fn setPixelsFromRectangle(this: *This, img: im.Img2D([4]u8), r: Rect) void {
    //     _ = cc.SDL_LockSurface(this.surface);

    //     const x_zoom = @intToFloat(f32, this.nx) / @intToFloat(f32, r.xmax - r.xmin);
    //     const y_zoom = @intToFloat(f32, this.ny) / @intToFloat(f32, r.ymax - r.ymin);

    //     for (this.pix.img, 0..) |*w, i| {
    //         const x_idx = r.xmin + divFloorIntByFloat(i % this.nx, x_zoom);
    //         const y_idx = r.ymin + divFloorIntByFloat(@divFloor(i, this.nx), y_zoom);
    //         const v = img.get(x_idx, y_idx).*;
    //         w.* = v;
    //     }
    //     cc.SDL_UnlockSurface(this.surface);
    // }
};
