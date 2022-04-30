const im = @import("imageBase.zig");
const std = @import("std");
const draw = @import("drawing_basic.zig");
const g = @import("geometry.zig");
const tg = @import("tri_grid.zig");

const Vec2 = g.Vec2;
const print = std.debug.print;

pub fn rasterize(self: g.Mesh2D, name: []const u8) !void {
    var pix = try im.Img2D([4]u8).init(width, width);
    defer pix.deinit();

    const bbox = g.boundsBBox(self.vs.items);
    const bbox_target = g.newBBox(10, width - 10, 10, width - 10);

    for (self.vs.items) |p| {
        const p2 = g.pt2PixCast(g.affine(p, bbox, bbox_target)); // transformed
        draw.drawCircle([4]u8, pix, p2[0], p2[1], 3, .{ 255, 255, 255, 255 });
    }

    var it = self.ts.keyIterator();

    while (it.next()) |_tri| {
        const tri = _tri.*;

        const p0 = g.pt2PixCast(g.affine(self.vs.items[tri[0]], bbox, bbox_target));
        const p1 = g.pt2PixCast(g.affine(self.vs.items[tri[1]], bbox, bbox_target));
        const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

        draw.drawLine(
            [4]u8,
            pix,
            p0[0],
            p0[1],
            p1[0],
            p1[1],
            .{ 255, 0, 0, 255 },
        );
        draw.drawLine(
            [4]u8,
            pix,
            p0[0],
            p0[1],
            p2[0],
            p2[1],
            .{ 255, 0, 0, 255 },
        );
        draw.drawLine(
            [4]u8,
            pix,
            p1[0],
            p1[1],
            p2[0],
            p2[1],
            .{ 255, 0, 0, 255 },
        );
    }

    try im.saveRGBA(pix, name);
}

const width = 1600;

pub fn rasterizeHighlightTri(self: g.Mesh2D, name: []const u8, tris: [][3]u32) !void {
    var pix = try im.Img2D([4]u8).init(width, width);
    defer pix.deinit();

    const bbox = g.boundsBBox(self.vs.items);
    const bbox_target = g.newBBox(10, width - 10, 10, width - 10);

    for (self.vs.items) |p| {
        const p2 = g.pt2PixCast(g.affine(p, bbox, bbox_target)); // transformed
        draw.drawCircle([4]u8, pix, p2[0], p2[1], 3, .{ 255, 255, 255, 255 });
    }

    var it = self.ts.keyIterator();

    while (it.next()) |_tri| {
        const tri = _tri.*;

        const p0 = g.pt2PixCast(g.affine(self.vs.items[tri[0]], bbox, bbox_target));
        const p1 = g.pt2PixCast(g.affine(self.vs.items[tri[1]], bbox, bbox_target));
        const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

        draw.drawLine(
            [4]u8,
            pix,
            p0[0],
            p0[1],
            p1[0],
            p1[1],
            .{ 255, 0, 0, 255 },
        );
        draw.drawLine(
            [4]u8,
            pix,
            p0[0],
            p0[1],
            p2[0],
            p2[1],
            .{ 255, 0, 0, 255 },
        );
        draw.drawLine(
            [4]u8,
            pix,
            p1[0],
            p1[1],
            p2[0],
            p2[1],
            .{ 255, 0, 0, 255 },
        );
    }

    for (tris) |tri| {

        // const tri = tris.*;

        const p0 = g.pt2PixCast(g.affine(self.vs.items[tri[0]], bbox, bbox_target));
        const p1 = g.pt2PixCast(g.affine(self.vs.items[tri[1]], bbox, bbox_target));
        const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

        draw.drawLine(
            [4]u8,
            pix,
            p0[0],
            p0[1],
            p1[0],
            p1[1],
            .{ 0, 0, 255, 255 },
        );
        draw.drawLine(
            [4]u8,
            pix,
            p0[0],
            p0[1],
            p2[0],
            p2[1],
            .{ 0, 0, 255, 255 },
        );
        draw.drawLine(
            [4]u8,
            pix,
            p1[0],
            p1[1],
            p2[0],
            p2[1],
            .{ 0, 0, 255, 255 },
        );
    }

    try im.saveRGBA(pix, name);
}

pub fn rasterizeHighlightStuff(self: g.Mesh2D, name: []const u8, pts: []Vec2, edges: [][2]u32, tris: [][3]u32) !void {
    var pix = try im.Img2D([4]u8).init(width, width);
    defer pix.deinit();

    const bbox = g.boundsBBox(self.vs.items);
    const bbox_target = g.newBBox(10, width - 10, 10, width - 10);

    for (self.vs.items) |p| {
        const p2 = g.pt2PixCast(g.affine(p, bbox, bbox_target)); // transformed
        draw.drawCircle([4]u8, pix, p2[0], p2[1], 3, .{ 255, 255, 255, 255 });
    }

    var it = self.ts.keyIterator();

    while (it.next()) |_tri| {
        const tri = _tri.*;

        const p0 = g.pt2PixCast(g.affine(self.vs.items[tri[0]], bbox, bbox_target));
        const p1 = g.pt2PixCast(g.affine(self.vs.items[tri[1]], bbox, bbox_target));
        const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

        draw.drawLine(
            [4]u8,
            pix,
            p0[0],
            p0[1],
            p1[0],
            p1[1],
            .{ 255, 0, 0, 255 },
        );
        draw.drawLine(
            [4]u8,
            pix,
            p0[0],
            p0[1],
            p2[0],
            p2[1],
            .{ 255, 0, 0, 255 },
        );
        draw.drawLine(
            [4]u8,
            pix,
            p1[0],
            p1[1],
            p2[0],
            p2[1],
            .{ 255, 0, 0, 255 },
        );
    }

    for (tris) |tri| {

        const p0 = g.pt2PixCast(g.affine(self.vs.items[tri[0]], bbox, bbox_target));
        const p1 = g.pt2PixCast(g.affine(self.vs.items[tri[1]], bbox, bbox_target));
        const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

        draw.drawLine(
            [4]u8,
            pix,
            p0[0],
            p0[1],
            p1[0],
            p1[1],
            .{ 0, 0, 255, 255 },
        );
        draw.drawLine(
            [4]u8,
            pix,
            p0[0],
            p0[1],
            p2[0],
            p2[1],
            .{ 0, 0, 255, 255 },
        );
        draw.drawLine(
            [4]u8,
            pix,
            p1[0],
            p1[1],
            p2[0],
            p2[1],
            .{ 0, 0, 255, 255 },
        );
    }

    for (edges) |edge| {

        const p0 = g.pt2PixCast(g.affine(self.vs.items[edge[0]], bbox, bbox_target));
        const p1 = g.pt2PixCast(g.affine(self.vs.items[edge[1]], bbox, bbox_target));
        // const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

        draw.drawLine(
            [4]u8,
            pix,
            p0[0],
            p0[1],
            p1[0],
            p1[1],
            .{ 0, 255, 0, 255 },
        );
    }

    for (pts) |p| {
        const p0 = g.pt2PixCast(g.affine(p, bbox, bbox_target));
        draw.drawCircle([4]u8, pix, p0[0], p0[1], 3, .{ 255, 0, 0, 255 });
    }

    try im.saveRGBA(pix, name);
}

// pub fn rasterizeTriGrid(trigrid:tg.GridHash2 , name: []const u8, pts: []Vec2, edges: [][2]u32, tris: [][3]u32) !void {

//     var pix = try im.Img2D([4]u8).init(width, width);
//     defer pix.deinit();

//     const bbox = g.boundsBBox(self.vs.items);
//     const bbox_target = g.newBBox(10, width - 10, 10, width - 10);

//     for (self.vs.items) |p| {
//         const p2 = g.pt2PixCast(g.affine(p, bbox, bbox_target)); // transformed
//         draw.drawCircle([4]u8, pix, p2[0], p2[1], 3, .{ 255, 255, 255, 255 });
//     }

//     var it = self.ts.keyIterator();

//     while (it.next()) |_tri| {
//         const tri = _tri.*;

//         const p0 = g.pt2PixCast(g.affine(self.vs.items[tri[0]], bbox, bbox_target));
//         const p1 = g.pt2PixCast(g.affine(self.vs.items[tri[1]], bbox, bbox_target));
//         const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

//         draw.drawLine(
//             [4]u8,
//             pix,
//             p0[0],
//             p0[1],
//             p1[0],
//             p1[1],
//             .{ 255, 0, 0, 255 },
//         );
//         draw.drawLine(
//             [4]u8,
//             pix,
//             p0[0],
//             p0[1],
//             p2[0],
//             p2[1],
//             .{ 255, 0, 0, 255 },
//         );
//         draw.drawLine(
//             [4]u8,
//             pix,
//             p1[0],
//             p1[1],
//             p2[0],
//             p2[1],
//             .{ 255, 0, 0, 255 },
//         );
//     }

//     for (tris) |tri| {

//         const p0 = g.pt2PixCast(g.affine(self.vs.items[tri[0]], bbox, bbox_target));
//         const p1 = g.pt2PixCast(g.affine(self.vs.items[tri[1]], bbox, bbox_target));
//         const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

//         draw.drawLine(
//             [4]u8,
//             pix,
//             p0[0],
//             p0[1],
//             p1[0],
//             p1[1],
//             .{ 0, 0, 255, 255 },
//         );
//         draw.drawLine(
//             [4]u8,
//             pix,
//             p0[0],
//             p0[1],
//             p2[0],
//             p2[1],
//             .{ 0, 0, 255, 255 },
//         );
//         draw.drawLine(
//             [4]u8,
//             pix,
//             p1[0],
//             p1[1],
//             p2[0],
//             p2[1],
//             .{ 0, 0, 255, 255 },
//         );
//     }

//     for (edges) |edge| {

//         const p0 = g.pt2PixCast(g.affine(self.vs.items[edge[0]], bbox, bbox_target));
//         const p1 = g.pt2PixCast(g.affine(self.vs.items[edge[1]], bbox, bbox_target));
//         // const p2 = g.pt2PixCast(g.affine(self.vs.items[tri[2]], bbox, bbox_target));

//         draw.drawLine(
//             [4]u8,
//             pix,
//             p0[0],
//             p0[1],
//             p1[0],
//             p1[1],
//             .{ 0, 255, 0, 255 },
//         );
//     }

//     for (pts) |p| {
//         const p0 = g.pt2PixCast(g.affine(p, bbox, bbox_target));
//         draw.drawCircle([4]u8, pix, p0[0], p0[1], 3, .{ 255, 0, 0, 255 });
//     }

//     try im.saveRGBA(pix, name);    
// }

pub fn main() !void {
    var a = std.testing.allocator;
    var the_mesh = try g.Mesh2D.initRand(a);
    defer the_mesh.deinit();

    the_mesh.show();
    try rasterize(the_mesh, "mesh0.tga");

    const tris = try the_mesh.validTris(a);
    defer a.free(tris);

    try the_mesh.removeTris(tris[0..1]);

    try rasterize(the_mesh, "mesh1.tga");

    the_mesh.show();

    const center_point = (the_mesh.vs.items[tris[1][0]] + the_mesh.vs.items[tris[1][1]] + the_mesh.vs.items[tris[1][2]]) / Vec2{ 3.0, 3.0 };

    print("cp {d}\n", .{center_point});
    print("tris1 = {d} \n", .{tris[1]});

    try the_mesh.addPointInPolygon(center_point, &tris[1]);

    the_mesh.show();

    try rasterize(the_mesh, "mesh2.tga");
}
