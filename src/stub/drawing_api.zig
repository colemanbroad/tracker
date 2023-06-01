// NOTE: To test if a realspace line intersects a grid box we CANT just take equally spaced samples along the line,
//     because we might skip a box if the intersection is very small.
//
//
//
//
//
//

// Create a Buffer, a Shape, DrawProperties (LineStyle). Then call `draw_xxx()`.
// Maybe also use an Affine mapping from shape coords ?
test "api 0" {

    // create and image (empty 2d buffer)
    var buf = Img2D(u8).init(1024, 1024);

    // create some shapes (circle, point, line, , rect, etc)
    const circle = Circle{ .center = .{ 9, 3 }, .radius = 3 };
    const line = Line{ .start = .{ 9, 3 }, .end = .{ 1, 2 } };
    const point = Pt{ 9, 3 };

    // draw shape into buffer with certain color, line width, etc
    drawShapeIntoBuffer(circle, buf, DrawProperties{ .color = .Red, .linewidth = 0.5 }); // option 1
    drawCircleIntoBuffer(circle, buf, .{ .color = .Red, .linewidth = 0.5 });
    drawCircleIntoBuffer(circle, buf, .Red, 0.5);

    const bezier = BezierCurve{ .interp = .ThirdOrder };
    drawBezierIntoBuffer(curve, buf, .{ .interp = 3 });

    const line = Line{ .start = Pt{ 1, 3.5 }, .end = Pt{ 2.0, 9 } };
    draw(buf, line, .Red, 0.3); // type dispatch on `line` goes to drawLine(buf,line,.Red,...) ?

    drawLine(buf, xy0, xy1, .Red, 2.5);
    drawCircle(buf, 9, .{ 5, 7 }, .Red, 2.5);

    // OR the drawing properties of `Circle` are contained within the type ?
    const newshape = .circle;
    draw2(buf, circle);
    draw2(buf, line);
    draw2(buf, box);
}

test "api 0 w affine" {
    // We create image buffers
    const img = try im.Img2D([4]u8).init(100, 101);
    // Create a shape
    const circle = Circle{ .center = .{ 9, 8 }, .radius = 5 };
    // Create line style
    const linestyle = LineStyle{ .width = 3, .color = .{ 100, 0, 255 }, .style = .dashed };
    // define affine mapping from continuous shape/line space to pixel space
    const affine = Affine{ .offset = .{ 2, 3 }, .scale = .{ 4, 4 } };
    // Draw shape into buffer using linestyle [ignore OOB pixels]
    draw(img, circle, linestyle, affine);
}

// Pass func ptr to e.g. `traceLineSegment()`
// We use the Shape to compute an iteration over pixels Pix
// Then we pass Pix into fnDrawAtEachPixel to actually perform draing.
// This func may have internal state to e.g. do dashed lines / expand.
// This API FAILS if the LineStyle could affect the Pix iteration...
// It also feels less direct and more yucky.
// But it is the most generic. Pure separation of drawing from getting pixel path.
test "api 2" {
    // We create image buffers
    const img = try im.Img2D([4]u8).init(100, 101);
    // Create a shape
    const circle = Circle{ .center = .{ 9, 8 }, .radius = 5 };
    // Design draw func. Color / linestyle / etc goes in here.
    // Each pixel that intersects circle is passed one time...
    fn fnDrawAtEachPixel(ctx: @TypeOf(ctx), pix: Vec2) void{};
    // perform the drawing
    traceCircleOutline(ctx, fnDrawAtEachPixel, circle);
}

// Turn shapes into Slices of pixel coordinates.
// Very simple. No draw calls. Probably
test "api 3" {
    // We create image buffers
    const img = try im.Img2D([4]u8).init(100, 101);
    // Create a shape
    const circle = Circle{ .center = .{ 9, 8 }, .radius = 5 };
    // Shape to Pixels
    const pixels = shape2Pixels(circle, .{ .fill = false }, trans_affine);
    // Do something with those pixels
    for (pixels) |p| img.getPix(p) = value;
}

/// STUB Impl
const shapes = struct {
    const Shape = enum {
        circle,
        square,
        box,
    };

    pub fn combine(sh1: Shape, sh2: Shape) Shape {
        _ = sh2;
        _ = sh1;
    }
    // pub fn add(sh1: Shape, sh2: Shape) Shape {}
};

fn draw2(buf: Img2D(u8), sh: Shape2) void {}
fn draw(buf: Img2D(u8), sh: Shape, lp: LineParams) void {}
fn drawLine(buf: Img2D(u8), xy0: Pt, xy1: Pt, c: Color, linewidth: f32) void {}
fn drawCircle(buf: Img2D(u8), r: f32, xy0: Pt, c: Color, linewidth: f32) void {}
fn drawCircles(buf: Img2D(u8), rs: []f32, xy0: [][2]f32, c: Color, linewidth: f32) void {}

const MarkStyle = enum { solid, dashed };
const BlendStyle = enum { add, overlay, multiply };
const Color = [4]u8;
const Affine = struct { offset = [2]f32, scale = [2]f32 };

/// TODO: End cap style? Anti Aliasing? Think of all the properties in Affinities LineStyle!
const Style = struct {
    width: f32,
    color: Color,
    style: MarkStyle = .solid,
    blend: BlendStyle = .overlay,
};

fn drawLine(buf: Img2D([4]u8), shape: LineSegment, ls: Style, aff: Affine) void {}
