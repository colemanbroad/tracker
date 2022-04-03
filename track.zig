// build with 
// zig build-lib track.zig -dynamic
// to make a dylib to load in python
// OR
// zig test track.zig

// zig run track.zig


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

// HOWTO: load 'lib.a' ?
// const del = @import("/Users/broaddus/Desktop/projects-personal/zig/zig-opencl-test/src/libdelaunay.a");

const del = @import("delaunay.zig");

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();


test "build delaunay neibs and dists" {
// pub fn main() !void {
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

  // get delaunay triangles
  const triangles = try del.delaunay2dal(allocator,randa[0..]);
  defer allocator.free(triangles);

  print("\n",.{});
  for (triangles) |t,i| {
    print("tri {} {d}\n", .{i,t});
    if (i>20) break;
  }

  // naive O(n^2) edge representation
  const edges = try allocator.alloc(u8, na*na);
  defer allocator.free(edges);
  for (edges) |*e| e.* = 0;
  for (triangles) |t| {
    edges[t[0]*na + t[1]] = 1;
    edges[t[1]*na + t[0]] = 1;
    edges[t[0]*na + t[2]] = 1;
    edges[t[2]*na + t[0]] = 1;
    edges[t[1]*na + t[2]] = 1;
    edges[t[2]*na + t[1]] = 1;
  }

  // Assignment Matrix. 0 = known negative. 1 = known positive. 2 = unknown.
  var asgn = try allocator.alloc(u8,na*nb);
  for (asgn) |*v| v.* = 2;

  // for each pair of vertices v0,v1 on a delaunay edge we have an associated cost for their translation difference
  // we also have a cost based on the displacement v0(t),v0(t+1)
  // we can pick a vertex at random and choose it's lowest cost match. then given that assignment we can fill in the rest.

  // find lowest cost edge for va=0 and add it to asgn (greedy)
  const best = blk: {
    var c_min = cost[0];
    var id:u32 = 0;
    for (cost[0..nb]) |c,i| {
      if (c<c_min) {
        c_min = c;
        id = i;
      }
    }
    break :blk id;
  };
  asgn[0*nb + best] = 1;
  
  // now look through the neighbours of va=0 and recompute their costs to include strain.
  // the strain cost only includes strain relative to neibs which have already been assigned (asgn=1) ?
  // or should we try to add the node with the smallest lower bound ? This is the really optimistic, greedy version.
  // We keep track of a lower bound for every assignment (which is the v(t)→ v(t+1) cost + best-case scenario for all neibs)
  // Then we can update this lower bound as we expand the solution.

  // I think I had always envisioned the solution to "spill" outwards from a single node. We would assign the nearest neighbours 
  // first, and then proceed outwards. This involves taking out Delaunay Graph and computing a distance tree to the solved node.
  // or, actually, it involves labeling nodes as "assigned", "unassigned", and "unassigned but with X assigned neighbours"...
  // Then we greedily look through all nodes which share the highest number of neighbour assignments and compute a best-case cost.
  // 

  // OH NO. We need to perform this search over vb (not va), and connect backwards to the parent?

  // In what order should we handle va's ?
  // Order the nodes by the number of neighbours which have already been assigned. Those are the least uncertain.
  // 

  // recompute neighbour assignment costs given this
  // 


  assert(false);


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




// pub fn main() !void {
test "greedy min-cost tracking" {

  print("\n\n",.{});

  const na = 1001;
  const nb = 1002;

  var va:[na]Pts = undefined;
  for (va) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10};
  var vb:[nb]Pts = undefined;
  for (vb) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10};

  _ = try greedy_track(va[0..],vb[0..]);
}




export fn greedy_track2d(va:[*]f32, na:u32 , vb:[*]f32, nb:u32) [*]i32 {

  const va_ = allocator.alloc(Pts,na) catch unreachable;
  for (va_) |*v,i| v.* = Pts{va[2*i], va[2*i+1]};
  const vb_ = allocator.alloc(Pts,nb) catch unreachable;
  for (vb_) |*v,i| v.* = Pts{vb[2*i], vb[2*i+1]};

  const parents = greedy_track(va_,vb_) catch unreachable;
  return parents.ptr;
}

pub fn greedy_track(va:[]Pts,vb:[]Pts) ![]i32 {

  const na = va.len;
  const nb = vb.len;

  // count number of out edges on A
  var aout = try allocator.alloc(u8, na);
  defer allocator.free(aout);
  for (aout) |*v| v.* = 0;

  // count number of in edges on B
  var bin = try allocator.alloc(u8, nb);
  defer allocator.free(bin);
  for (bin) |*v| v.* = 0;

  var cost = try allocator.alloc(f32,na*nb);
  defer allocator.free(cost);
  for (cost) |*v| v.* = 0;
  pairwise_distances(cost,va[0..],vb[0..]);

  var asgn = try allocator.alloc(u8,na*nb);
  defer allocator.free(asgn);
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


  // greedily go through edges and add them to graph iff they don't violate constraints
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
  
  }

  var parents = try allocator.alloc(i32,nb);
  for (vb) |_,i| {
    for (va) |_,j| {
      // if the edge is active, then assign and break (can only have 1 parent max)
      if (asgn[j*nb+i]==1) {
        parents[i] = @intCast(i32,j);
        break;
      }
      // otherwise there is no parent
      parents[i] = -1;
    }
  }

  for (parents) |p,i| {
    print("{d} → {d}\n", .{i,p});
    if (i>10) break;
  }

  return parents;

  // for (va) |_,i| {
  //   print("{d}\n", .{asgn[i*nb..(i+1)*nb]});
  //   if (i>10) break;
  // }
}





// Zig doesn't have tuple-of-types i.e. product types yet so all fields must be named. This is probably good.
// https://github.com/ziglang/zig/issues/4335 
const CostEdgePair:type = struct{cost:f32, ia:u32, ib:u32};
const testing = std.testing;

// var allocator = std.heap.GeneralPurposeAllocator(.{}){};
var allocator = std.testing.allocator; //(.{}){};

// const Order = std.math.Order;
fn lessThan(context: void, a: CostEdgePair, b: CostEdgePair) std.math.Order {
  _ = context;
  return std.math.order(a.cost, b.cost);
}





// const Pts = [3]f32;
const Pts = del.Vec2;


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







