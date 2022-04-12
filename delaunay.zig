const std = @import("std");
const geo = @import("geometry.zig");
pub const Vec2 = geo.Vec2;
const max = std.math.max;
const min = std.math.min;

// const spatial = @import("spatial.zig");
const print = std.debug.print;

// var gpa = std.heap.GeneralPurposeAllocator(.{}){};
// const alloc = gpa.allocator();
const alloc = std.testing.allocator;

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();

test {std.testing.refAllDecls(@This());}

test "resize a thing" {
  var pts = try alloc.alloc(f32, 100);
  defer alloc.free(pts);
  pts = alloc.resize(pts, pts.len + 1).?;
}

test "resize many times" {
  var count: u32 = 100;
  while (count < 10000) : (count += 100) {
    var pts = try alloc.alloc([2]f32, count);
    defer alloc.free(pts);

    for (pts) |*v, j| {
      const i = @intToFloat(f32, j);
      v.* = .{ i, i * i };
    }

    try testresize(pts);
    print("after resize...\npts = {d}\n", .{pts[pts.len - 3 ..]});
    print("len={d}", .{pts.len});
  }
}

fn testresize(_pts: [][2]f32) !void {
  var pts = alloc.resize(_pts, _pts.len + 3).?;
  defer _ = alloc.shrink(pts, pts.len - 3);
  pts[pts.len - 3] = .{ 1, 0 };
  pts[pts.len - 2] = .{ 1, 1 };
  pts[pts.len - 1] = .{ 1, 2 };
  print("pts[-3..] = {d}\n", .{pts[pts.len - 3 ..]});
  print("in testresize() : len={d}", .{pts.len});
}

pub fn delaunay2d(_pts: []Vec2) ![][3]u32 {
  return delaunay2dal(alloc, _pts);
}

fn lessThanDist(p0: Vec2, p_l: Vec2, p_r: Vec2) bool {
    // if (1 > 0) return true else return false;
    const dx_l = p0 - p_l;
    const d_l = @sqrt(@reduce(.Add, dx_l * dx_l));
    const dx_r = p0 - p_r;
    const d_r = @sqrt(@reduce(.Add, dx_r * dx_r));
    if (d_l < d_r) return true else return false;
  }

// Implementation of Bowyer-Watson algorithm for 2D tessellations
// SORTS _PTS IN PLACE !
pub fn delaunay2dal(allo: std.mem.Allocator, _pts: []Vec2) ![][3]u32 {

    // pts.len += 3;
    // pts = try gpa.allocator.resize(pts, pts.len+3);
    // defer _ = gpa.a
    // defer _ = gpa.allocator.shrink(pts,pts.len-3); // we know we will succeed. `defer` does not allow failure.
    // defer gpa.allocator.free(pts);
    // NOTE: using `_ = try realloc(...)` we can't use `defer` and so must place this line at end of function.

    // const pts = try gpa.allocator.realloc(_pts,_pts.len+3);

    const bbox = geo.boundsBBox(_pts);
    const box_width = bbox.x.hi - bbox.x.lo;
    const box_height = bbox.y.hi - bbox.y.lo;

    // TODO: change iteration order s.t. nearby points are first! This should reduce the number of bad triangles.

    // const sqrtN = @floatToInt(u16,@sqrt(@intToFloat(f32,N)));
    // var gh = try GridHash.init(allocator,sqrtN,sqrtN,20,pts[0..],null);
    // defer gh.deinit();

    // // Prealloc idx and dist memory for fast multiple queries
    // TODO: Radix Sort `pts` on x,y coordinates (bin space s.t. nearby points are close together. maybe use wider bin along the slow dimension for better locality?)
    // OR better idea, just sort by distance to a given point (could even take point that's OUTSIDE the set, i.e. mean)
    // Then procede outwards.

    // const res = try IdsDists.init(allocator,200);
    // defer res.deinit(allocator);

    // sorting the boundary points doesn't seem to help speed things up.
    // it just adds time... i guess if i wanted to speed things up i wouldn't
    // compare points against ALL triangles, but somehow...
    std.sort.sort(Vec2, _pts, Vec2{ bbox.x.lo, bbox.y.lo }, lessThanDist);

    var pts = try allo.alloc(Vec2, _pts.len);
    defer allo.free(pts);
    for (pts) |*p, i| p.* = _pts[i];

    // const boundaryVert1 = @intCast(u32, pts.len - 3);
    // const boundaryVert2 = @intCast(u32, pts.len - 2);
    // const boundaryVert3 = @intCast(u32, pts.len - 1);

    // pick points that form a triangle around the bounding box but don't go too wide
    const oldlen = @intCast(u32, pts.len);
    pts = try allo.realloc(pts, oldlen + 3);
    pts[oldlen + 0] = .{ bbox.x.lo - box_width * 0.1, bbox.y.lo - box_height * 0.1 }; //.{-1e5,-1e5};
    pts[oldlen + 1] = .{ bbox.x.lo - box_width * 0.1, bbox.y.hi + 2 * box_height }; // .{-1e5,1e5};
    pts[oldlen + 2] = .{ bbox.x.hi + 2 * box_width, bbox.y.lo - box_height * 0.1 }; //.{1e5,0};

    // pts is now FIXED. The memory doesn't change,

    // must store "maybe" triangles so we can easily remove triangles by setting them to null.
    var triangles = try std.ArrayList(?[3]u32).initCapacity(allo, pts.len * 100);
    defer triangles.deinit();

    // defer gpa.allocator.free(triangles);
    // @compileLog(@TypeOf(triangles));
    triangles.appendAssumeCapacity([3]u32{ oldlen + 0, oldlen + 1, oldlen + 2 });

    // holds invalid triangles that fail the circle test. indexes into `triangles`
    var badtriangles = try std.ArrayList(u32).initCapacity(allo, pts.len);
    defer badtriangles.deinit();

    // holds unique edge polygon border. indexes into `pts`
    var polyedges = try std.ArrayList([2]u32).initCapacity(allo, pts.len);
    defer polyedges.deinit();

    // if bad triangle edge is unique, then add it to big polygon.
    // First count the number of occurrences of each edge. Then add edges with count==1 to polygon.
    var edgehash = std.AutoHashMap([2]u32, u8).init(allo);
    defer edgehash.deinit();

    // MAIN LOOP OVER (nonboundary) POINTS
    for (pts[0 .. pts.len - 3]) |p, idx_pt| {

        // if pt in triangle, then triangle is bad.
        badtriangles.clearRetainingCapacity();

        // TODO speed up by only looping over nearby triangles.
        // how can we _prove_ that a set of triangles are invalid?

        for (triangles.items) |tri, tri_idx| {
          const vtri = if (tri) |v| v else continue;
            // if (tri) |vtri| {
              const tripts = [3]Vec2{ pts[vtri[0]], pts[vtri[1]], pts[vtri[2]] };
            // const delta = tripts[0] - p;
            // if (dot2(delta,delta)>2500) continue; // TODO: Can we make this algorithm more efficient by assuming a certain point density?
            if (geo.pointInTriangleCircumcircle2d(p, tripts)) {
              try badtriangles.append(@intCast(u32, tri_idx));
            }
            // }
          }

        // clear the map, but don't release the memory.
        edgehash.clearRetainingCapacity();

        // count the number of occurrences of each edge in bad triangles. unique edges occur once.
        for (badtriangles.items) |tri_idx| {
            const tri = triangles.items[tri_idx].?; // we know tri_idx only refers to valid,bad triangles
            const v0 = tri[0];
            const v1 = tri[1];
            const v2 = tri[2];
            const e0 = .{ min(v0, v1), max(v0, v1) };
            const e1 = .{ min(v1, v2), max(v1, v2) };
            const e2 = .{ min(v2, v0), max(v2, v0) };
            if (edgehash.get(e0)) |c| {
              try edgehash.put(e0, c + 1);
              } else try edgehash.put(e0, 1);
              if (edgehash.get(e1)) |c| {
                try edgehash.put(e1, c + 1);
                } else try edgehash.put(e1, 1);
                if (edgehash.get(e2)) |c| {
                  try edgehash.put(e2, c + 1);
                  } else try edgehash.put(e2, 1);
                }

        // edges that occur once are added to polyedges
        polyedges.clearRetainingCapacity();
        for (badtriangles.items) |tri_idx| {
          const tri = triangles.items[tri_idx].?;
          const v0 = tri[0];
          const v1 = tri[1];
          const v2 = tri[2];
          const e0 = .{ min(v0, v1), max(v0, v1) };
          const e1 = .{ min(v1, v2), max(v1, v2) };
          const e2 = .{ min(v2, v0), max(v2, v0) };
          const c0 = edgehash.get(e0);
          if (c0.? == 1) {
            try polyedges.append(e0);
          }
          const c1 = edgehash.get(e1);
          if (c1.? == 1) {
            try polyedges.append(e1);
          }
          const c2 = edgehash.get(e2);
          if (c2.? == 1) {
            try polyedges.append(e2);
          }
        }

        // Remove bad triangles by setting to null
        // NOTE: we save time by avoiding the need to compress arraylist, but it wastes lots of space!
        for (badtriangles.items) |tri_idx| {
          triangles.items[tri_idx] = null;
        }

        for (polyedges.items) |edge| {
          try triangles.append([3]u32{ edge[0], edge[1], @intCast(u32, idx_pt) });
            // triangles.appendAssumeCapacity([3]u32{edge[0],edge[1],@intCast(u32,idx_pt)});
            // triangles.appendAssumeCapacity([3]u32{edge[0],edge[1],@intCast(u32,idx_pt)});
          }

        // if (@intToFloat(f32, triangles.items.len) > @intToFloat(f32,triangles.capacity) * 0.9) triangles.items = removeNullFromList(?[3]u32, triangles.items);

        // try showdelaunaystate(pts,triangles,idx_pt); // save to image
        // print("lengths of things : {d}\n", .{idx_pt});
        // print("triangles {d}\n", .{triangles.items.len});
        // print("badtriangles {d}\n", .{badtriangles.items.len});
        // print("polyedges {d}\n", .{polyedges.items.len});

        // if (try checkDelaunay(idx_pt,pts,triangles.items)) @breakpoint();
      }

    // clean up
    var idx_valid: u32 = 0;
    var validtriangles = try allo.alloc([3]u32, triangles.items.len);

    for (triangles.items) |tri| {
        // valid_triangle
        if (tri) |vt| {
            // remove triangles containing starting points
            if (vt[0] >= oldlen or vt[1] >= oldlen or vt[2] >= oldlen) continue;
            validtriangles[idx_valid] = vt;
            idx_valid += 1;
          }
        }

    // _ = try allo.realloc(pts,pts.len-3); // FREE extra points
    // validtriangles = validtriangles[0..idx_valid];
    validtriangles = try allo.realloc(validtriangles, idx_valid);
    // print("There were {d} valid out of / {d} total triangles. validtriangles has len={d}.\n", .{idx_valid, triangles.items.len, validtriangles.len});

    return validtriangles;
  }

  pub fn removeNullFromList(comptime T: type, arr: []T) []T {
    var idx: usize = 0;
    for (arr) |v| {
      if (v != null) {
        arr[idx] = v;
        idx += 1;
      }
    }
    return arr[0..idx];
  }

  const process = std.process;

  test "basic delaunay" {
  // pub fn main() !void {

    const nparticles = if (@import("builtin").is_test) 100 else blk: {
      var arg_it = try process.argsWithAllocator(alloc);
      _ = arg_it.skip(); // skip exe name
      const npts_str = arg_it.next() orelse "100";
      break :blk try std.fmt.parseUnsigned(usize, npts_str, 10);
    };

    // 10k requires too much memory for triangles. The scaling is nonlinear.
    // After change I can do 10k (0.9s) 20k (2.7s) 30k (6s) 40k (10s) ...
    const verts = try alloc.alloc(Vec2, nparticles);
    for (verts) |*v| v.* = .{ random.float(f32), random.float(f32) };
    defer alloc.free(verts); // changes size when we call delaunay2d() ...

    const triangles = try delaunay2d(verts);
    defer alloc.free(triangles);

    print("\n\nfound {} triangles on {} vertices\n", .{ triangles.len, nparticles });
  }

// how to iterate over points starting from a random `p` and proceding to it's neighbours.
// The order doesn't really matter so much. Just pick a random point and sort by distance to that point. That's N log N.

