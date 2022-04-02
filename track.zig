// build with 
// zig build-lib test_zig.zig -dynamic
// to make a dylib to load in python
// OR
// zig test test_zig.zig


const std = @import("std");
const PriorityQueue = std.PriorityQueue;
const print = std.debug.print;
const assert = std.debug.assert;

const T = i32;

export fn add(a : T, b : T) T {
  return a + b;
}

export fn sum(a : [*]T, n : u32) T {
  var tot:T = 0;
  var i:u32 = 0;
  while (i<n) {
    tot += a[i];
    i += 1;
    // print("Print {} me {} you fool!\n", .{tot,i});
  }
  return tot;
}

// a:[*]T, b:[*]T, na:u32, nb:u32

// const del = @import("/Users/broaddus/Desktop/projects-personal/zig/zig-opencl-test/src/libdelaunay.a");
const del = @import("delaunay.zig");

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();


// test "build delaunay neibs and dists" {
pub fn main() !void {
  print("\n\n",.{});

  const na = 2001;
  const nb = 2002;
  // count number of out edges on A
  var aout:[na]u8 = undefined;
  for (aout) |*v| v.* = 0;
  var randa:[na]Pts = undefined;
  // for (randa) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10, random.float(f32)*10};
  for (randa) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10};

  // count number of in edges on B
  var bin:[nb]u8 = undefined;
  for (bin) |*v| v.* = 0;
  var randb:[nb]Pts = undefined;
  // for (randb) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10, random.float(f32)*10};
  for (randb) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10};

  // var cost:[na*nb]f32 = undefined;
  var cost = try allocator.alloc(f32,na*nb);
  for (cost) |*v| v.* = 0;
  pairwise_distances(cost,randa[0..],randb[0..]);


  // const verts = try alloc.alloc(Vec2, na); 

  // for (verts) |*v,i| v.* = 
  // defer alloc.free(verts); // changes size when we call delaunay2d() ... 

  const triangles = try del.delaunay2dal(allocator,randa[0..]);
  defer allocator.free(triangles);

  print("\n",.{});

  for (triangles) |t,i| {
    print("tri {} {d}\n", .{i,t});
    if (i>20) break;
  }

  assert(false);

  // print distances (costs)
  // for (randa) |_,i| {
  //   print("{d:.3}\n", .{cost[i*nb..(i+1)*nb]});
  // }


  // var asgn:[na*nb]u8 = undefined;
  var asgn = try allocator.alloc(u8,na*nb);
  for (asgn) |*v| v.* = 0;

  // sort costs
  // continue adding costs cheapest-first as long as they don't violate asgn constraints
  // sort each row by smallest cost? then 

  const PQlt = PriorityQueue(CostEdgePair, void, lessThan);
  var queue = PQlt.init(allocator, {});
  defer queue.deinit();
  for (cost) |c,i| {
    const ia = i / nb;
    const ib = i % nb;
    try queue.add(.{.cost=c,.ia=@intCast(u32,ia),.ib=@intCast(u32,ib)});
  }

  // greedily go through edges and add them graph iff they don't violate constraints

  var count:usize = 0;
  while (true) {

    count += 1;
    if (count == na*nb + 1) break;

    const edge = queue.remove();
    // print("Cost {}, Edge {} {}\n", edge);

    if (aout[edge.ia]==2 or bin[edge.ib]==1) continue;
    asgn[edge.ia*nb + edge.ib] = 1;
    
    aout[edge.ia] += 1;
    bin[edge.ib] += 1;
    
    // if (count > 3*na/2) break;
  }

  for (randa) |_,i| {
    print("{d}\n", .{asgn[i*nb..(i+1)*nb]});
    if (i>10) break;
  }
}


// test "greedy min-cost tracking" {
// // pub fn main() !void {

//   print("\n\n",.{});

//   // const v2 = del.Vec2{1,9};
//   // print("v2 = {}", .{v2});

//   const na = 2001;
//   const nb = 2002;
//   // count number of out edges on A
//   var aout:[na]u8 = undefined;
//   for (aout) |*v| v.* = 0;
//   var randa:[na]Pts = undefined;
//   for (randa) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10, random.float(f32)*10};

//   // count number of in edges on B
//   var bin:[nb]u8 = undefined;
//   for (bin) |*v| v.* = 0;
//   var randb:[nb]Pts = undefined;
//   for (randb) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10, random.float(f32)*10};

//   // var cost:[na*nb]f32 = undefined;
//   var cost = try allocator.alloc(f32,na*nb);
//   for (cost) |*v| v.* = 0;
//   pairwise_distances(cost,randa[0..],randb[0..]);

//   // print distances (costs)
//   // for (randa) |_,i| {
//   //   print("{d:.3}\n", .{cost[i*nb..(i+1)*nb]});
//   // }

//   // var asgn:[na*nb]u8 = undefined;
//   var asgn = try allocator.alloc(u8,na*nb);
//   for (asgn) |*v| v.* = 0;

//   // sort costs
//   // continue adding costs cheapest-first as long as they don't violate asgn constraints
//   // sort each row by smallest cost? then 

//   const PQlt = PriorityQueue(CostEdgePair, void, lessThan);
//   var queue = PQlt.init(allocator, {});
//   defer queue.deinit();
//   for (cost) |c,i| {
//     const ia = i / nb;
//     const ib = i % nb;
//     try queue.add(.{.cost=c,.ia=@intCast(u32,ia),.ib=@intCast(u32,ib)});
//   }

//   // greedily go through edges and add them graph iff they don't violate constraints

//   var count:usize = 0;
//   while (true) {

//     count += 1;
//     if (count == na*nb + 1) break;

//     const edge = queue.remove();
//     // print("Cost {}, Edge {} {}\n", edge);

//     if (aout[edge.ia]==2 or bin[edge.ib]==1) continue;
//     asgn[edge.ia*nb + edge.ib] = 1;
    
//     aout[edge.ia] += 1;
//     bin[edge.ib] += 1;
    
//     // if (count > 3*na/2) break;
//   }

//   for (randa) |_,i| {
//     print("{d}\n", .{asgn[i*nb..(i+1)*nb]});
//     if (i>10) break;
//   }
// }






// const allocator = testing.Allocator;
const CostEdgePair:type = struct{cost:f32, ia:u32, ib:u32};
const testing = std.testing;
// const allocator = std.mem.Allocator;
// const allocator = std.mem.Allocator;
// var allocator = std.heap.GeneralPurposeAllocator(.{}){};
var allocator = std.testing.allocator; //(.{}){};
const Order = std.math.Order;

fn lessThan(context: void, a: CostEdgePair, b: CostEdgePair) Order {
  _ = context;
  return std.math.order(a.cost, b.cost);
}



// const Pts = [3]f32;
const Pts = del.Vec2;

// fn greedytrack(a:[]Pts, b:[]Pts) void {
//   const pd = pairwise_distances();
// }

// array order is [a,b]. i.e. a has stride nb. b has stride 1.
fn pairwise_distances(cost:[]f32, a:[]Pts, b:[]Pts) void {
  const na = a.len;
  const nb = b.len;
  assert(cost.len == na*nb);
  for (a) |x,i| {
    for (b) |y,j| {
      cost[i*nb + j] = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]);
      // cost[i*nb + j] = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2]);
    }
  }
}