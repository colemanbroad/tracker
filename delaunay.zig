const std = @import("std");
const g = @import("./geometry.zig");
pub const Vec2 = g.Vec2;
const max = std.math.max;
const min = std.math.min;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const alloc = gpa.allocator();


var prng = std.rand.DefaultPrng.init(0);
const print = std.debug.print;

var globalidx:u32 = 0;

test "resize a thing" {
  var pts = try alloc.alloc(f32,100);
  pts = alloc.resize(pts, pts.len+1).?;
}

test "resize many times" {

  var count:u32=100;
  while (count<10000) : (count+=100) {
    var pts = try alloc.alloc([2]f32, count);
    defer alloc.free(pts);

    for (pts) |*v,j| {
      const i = @intToFloat(f32,j);
      v.* = .{i,i*i};
    }

    try testresize(pts);
    print("after resize...\npts = {d}\n", .{pts[pts.len-3..]});
    print("len={d}",.{pts.len});
  }
}

fn testresize(_pts:[][2]f32) !void {
  var pts = alloc.resize(_pts,_pts.len + 3).?;
  defer _ = alloc.shrink(pts, pts.len-3);
  pts[pts.len-3] = .{1,0};
  pts[pts.len-2] = .{1,1};
  pts[pts.len-1] = .{1,2};
  print("pts[-3..] = {d}\n",.{pts[pts.len-3..]});
  print("in testresize() : len={d}",.{pts.len});
}


pub fn delaunay2d(_pts:[]Vec2) ![][3]u32 {
  return delaunay2dal(alloc, _pts);
}


// Implementation of Bowyer-Watson algorithm for 2D tessellations
pub fn delaunay2dal(allo:std.mem.Allocator, _pts:[]Vec2) ![][3]u32 {

  // const pts = try gpa.allocator.realloc(_pts,_pts.len+3);
  var pts = try allo.alloc(Vec2, _pts.len+3);
  for (_pts) |p,i| pts[i] = p;
  // pts.len += 3;
  // pts = try gpa.allocator.resize(pts, pts.len+3);
  // defer _ = gpa.a
  // defer _ = gpa.allocator.shrink(pts,pts.len-3); // we know we will succeed. `defer` does not allow failure.
  // defer gpa.allocator.free(pts);
  // NOTE: using `_ = try realloc(...)` we can't use `defer` and so must place this line at end of function.
  
  const boundaryVert1 = @intCast(u32, pts.len - 3);
  const boundaryVert2 = @intCast(u32, pts.len - 2);
  const boundaryVert3 = @intCast(u32, pts.len - 1);
  pts[boundaryVert1] = .{-1e5,-1e5};
  pts[boundaryVert2] = .{-1e5,1e5};
  pts[boundaryVert3] = .{1e5,0};

  // pts is now FIXED. The memory doesn't change,

  // must store "maybe" triangles so we can easily remove triangles by setting them to null.
  var triangles = try std.ArrayList(?[3]u32).initCapacity(allo , pts.len*100);
  defer triangles.deinit();

  // defer gpa.allocator.free(triangles);
  // @compileLog(@TypeOf(triangles));
  triangles.appendAssumeCapacity([3]u32{boundaryVert1,boundaryVert2,boundaryVert3});

  // holds invalid triangles that fail the circle test. indexes into `triangles`
  var badtriangles = try std.ArrayList(u32).initCapacity(allo , pts.len);
  defer badtriangles.deinit();

  // holds unique edge polygon border. indexes into `pts`
  var polyedges = try std.ArrayList([2]u32).initCapacity(allo , pts.len);
  defer polyedges.deinit();


  // MAIN LOOP OVER (nonboundary) POINTS
  for (pts[0..pts.len-3]) |p,idx_pt| {

    // if pt in triangle, then triangle is bad.
    badtriangles.clearRetainingCapacity();
    for (triangles.items) |tri , tri_idx| {
      if (tri) |vtri| {
        const tripts = [3]Vec2{pts[vtri[0]] , pts[vtri[1]] , pts[vtri[2]]};
        // const delta = tripts[0] - p;
        // if (dot2(delta,delta)>2500) continue; // TODO: Can we make this algorithm more efficient by assuming a certain point density?
        if (g.pointInTriangleCircumcircle2d(p,tripts)) badtriangles.appendAssumeCapacity(@intCast(u32,tri_idx));        
      }
    }
    
    // if bad triangle edge is unique, then add it to big polygon.
    // First count the number of occurrences of each edge. Then add edges with count==1 to polygon.
    var edgehash = std.AutoHashMap([2]u32,u8).init(allo); // TODO: move allocation out of loop.
    defer edgehash.deinit();

    // count the number of occurrences of each edge in bad triangles. unique edges occur once.
    for (badtriangles.items) |tri_idx| {
      const tri = triangles.items[tri_idx].?; // we know tri_idx only refers to valid,bad triangles
      const v0 = tri[0];
      const v1 = tri[1];
      const v2 = tri[2];
      const e0 = .{min(v0,v1) , max(v0,v1)};
      const e1 = .{min(v1,v2) , max(v1,v2)};
      const e2 = .{min(v2,v0) , max(v2,v0)};
      if (edgehash.get(e0)) |c| {try edgehash.put(e0,c+1);} else try edgehash.put(e0,1);
      if (edgehash.get(e1)) |c| {try edgehash.put(e1,c+1);} else try edgehash.put(e1,1);
      if (edgehash.get(e2)) |c| {try edgehash.put(e2,c+1);} else try edgehash.put(e2,1);
    }

    // edges that occur once are added to polyedges
    polyedges.clearRetainingCapacity();
    for (badtriangles.items) |tri_idx| {
      const tri = triangles.items[tri_idx].?;
      const v0 = tri[0];
      const v1 = tri[1];
      const v2 = tri[2];
      const e0 = .{min(v0,v1) , max(v0,v1)};
      const e1 = .{min(v1,v2) , max(v1,v2)};
      const e2 = .{min(v2,v0) , max(v2,v0)};
      const c0 = edgehash.get(e0); if (c0.?==1) {polyedges.appendAssumeCapacity(e0);}
      const c1 = edgehash.get(e1); if (c1.?==1) {polyedges.appendAssumeCapacity(e1);}
      const c2 = edgehash.get(e2); if (c2.?==1) {polyedges.appendAssumeCapacity(e2);}
    }

    // Remove bad triangles by setting to null
    // NOTE: we save time by avoiding the need to compress arraylist, but it wastes lots of space!
    for (badtriangles.items) |tri_idx| {
      triangles.items[tri_idx] = null;
    }

    for (polyedges.items) |edge| {
      triangles.appendAssumeCapacity([3]u32{edge[0],edge[1],@intCast(u32,idx_pt)});
      // triangles.appendAssumeCapacity([3]u32{edge[0],edge[1],@intCast(u32,idx_pt)});
    }


    // if (@intToFloat(f32, triangles.items.len) > @intToFloat(f32,triangles.capacity) * 0.9) triangles.items = removeNullFromList(?[3]u32, triangles.items);

    // try showdelaunaystate(pts,triangles,idx_pt); // save to image
    print("lengths of things : {d}\n", .{idx_pt});
    print("triangles {d}\n", .{triangles.items.len});
    print("badtriangles {d}\n", .{badtriangles.items.len});
    print("polyedges {d}\n", .{polyedges.items.len});
  
    // if (try checkDelaunay(idx_pt,pts,triangles.items)) @breakpoint();
  }

  // clean up
  var idx_valid:u32 = 0;
  var validtriangles = try allo.alloc([3]u32 , triangles.items.len);

  for (triangles.items) |tri| {
    if (tri) |vt| { // valid_triangle
      if (vt[0]>=boundaryVert1 or vt[1]>=boundaryVert1 or vt[2]>=boundaryVert1) continue; // remove triangles containing starting points
      validtriangles[idx_valid] = vt;
      idx_valid += 1;
    }
  }



  // _ = try allo.realloc(pts,pts.len-3); // FREE extra points
  // validtriangles = validtriangles[0..idx_valid];
  validtriangles = try allo.realloc(validtriangles,idx_valid);
  print("There were {d} valid out of / {d} total triangles. validtriangles has len={d}.\n", .{idx_valid, triangles.items.len, validtriangles.len});

  return validtriangles;
}

pub fn removeNullFromList(comptime T:type , arr:[]T) []T {
  var idx:usize = 0;
  for (arr) |v| {
    if (v!=null) {
      arr[idx] = v;
      idx += 1;
    }
  }
  return arr[0..idx];
}

test "basic delaunay" {
  const nparticles = 1000;
  // 10k requires too much memory for triangles. The scaling is nonlinear.
  const verts = try alloc.alloc(Vec2, nparticles); 
  defer alloc.free(verts); // changes size when we call delaunay2d() ... 

  const triangles = try delaunay2d(verts);
  defer alloc.free(triangles);

  print("\n\nfound {} triangles on {} vertices\n", .{triangles.len, nparticles});
}


