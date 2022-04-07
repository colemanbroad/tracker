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

const NT = i32;
export fn add(a : NT, b : NT) NT {
  return a + b;
}

export fn sum(a : [*]NT, n : u32) NT {
  var tot:NT = 0;
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


// test "strain tracking" {
pub fn main() !void {
  print("\n\n",.{});

  const na = 1001;
  const nb = 1002;

  var va:[na]Pts = undefined;
  for (va) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10};
  var vb:[nb]Pts = undefined;
  for (vb) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10};

  const b2a = try strain_track(va[0..],vb[0..]);
  defer allocator.free(b2a);

  var res:[nb]i32 = undefined;

  for (b2a) |a_idx,i| {
    if (a_idx) |id| {
      res[i] = @intCast(i32,id);
    } else {
      res[i] = -1;
    }
    print("{}→{}\n",.{i,res[i]});
  }
}


const VStatusTag = enum(u3) {
  unknown,
  parent,
  // divider, // TODO
  daughter,
  appear,
  disappear,
};

export fn strain_track2d(va:[*]f32, na:u32 , vb:[*]f32, nb:u32, res:[*]i32) i32 {

  const va_ = allocator.alloc(Pts,na) catch return -1;
  for (va_) |*v,i| v.* = Pts{va[2*i], va[2*i+1]};
  const vb_ = allocator.alloc(Pts,nb) catch return -1;
  for (vb_) |*v,i| v.* = Pts{vb[2*i], vb[2*i+1]};

  const b2a = strain_track(va_,vb_) catch return -1;
  defer allocator.free(b2a);

  // Write the result to RES in-place
  for (b2a) |a_idx,i| {
    if (a_idx) |id| {
      res[i] = @intCast(i32,id);
    } else {
      res[i] = -1;
    }
  }

  return 0;
}

pub fn strain_track(va:[]Pts, vb:[]Pts) ![]?u32 {

  const na = va.len;
  const nb = vb.len;

  var a_status = try allocator.alloc(VStatusTag,na);
  defer allocator.free(a_status);
  for (a_status) |*v| v.* = .unknown;

  var b_status = try allocator.alloc(VStatusTag,nb);
  defer allocator.free(b_status);
  for (b_status) |*v| v.* = .unknown;

  var cost = try allocator.alloc(f32,na*nb);
  defer allocator.free(cost);
  for (cost) |*v| v.* = 0;
  pairwise_distances(Pts,cost,va[0..],vb[0..]);

  // get delaunay triangles
  // FIXME 2D only for now
  const triangles = try del.delaunay2dal(allocator,va[0..]);
  defer allocator.free(triangles);

  // print("\n",.{});
  // for (triangles) |t,i| {
  //   print("tri {} {d}\n", .{i,t});
  //   if (i>20) break;
  // }

  // Assignment Matrix. 0 = known negative. 1 = known positive. 2 = unknown.
  // var asgn = try allocator.alloc(u8,na*nb);
  // for (asgn) |*v| v.* = 2;

  var asgn_a2b = try allocator.alloc([2]?u32,na);
  defer allocator.free(asgn_a2b);
  for (asgn_a2b) |*v| v.* = .{null,null};

  var asgn_b2a = try allocator.alloc(?u32,nb);
  // defer allocator.free(asgn_b2a); // RETURNED BY FUNC
  for (asgn_b2a) |*v| v.* = null;

  // Count delaunay neighbours which are known, either .parent or .disappear
  var va_neib_count = try allocator.alloc(u8,na);
  defer allocator.free(va_neib_count);
  for (va_neib_count) |*v| v.* = 0;

  // 
  var delaunay_array = try allocator.alloc([8]?u32,na);
  defer allocator.free(delaunay_array);
  for (delaunay_array) |*v| v.* = .{null}**8; // id, id, id, id, ...

  var nn_distance_ditribution = try allocator.alloc([8]?f32,na);
  defer allocator.free(nn_distance_ditribution);
  for (nn_distance_ditribution) |*v| v.* = .{null}**8; // id, id, id, id, ...

  // Be careful. We're iterating through triangles, so we see each interior edge TWICE!
  // This loop will de-duplicate edges.
  // for each vertex `v` from triangle `tri` with neighbour vertex `v_neib` we loop over 
  // all existing neibs to see if `v_neib` already exists. if it doesn't we add it.
  // WARNING: this will break early once we hit `null`. This is fine as long as the array is front-packed like
  // [value value value null null ...]

  for (triangles) |tri| {
    for (tri) |v,i| {
      outer: for ([3]u32{0,1,2}) |j| {
        if (i==j) continue;
        const v_neib = tri[j];
        for (delaunay_array[v]) |v_neib_existing,k| {
          if (v_neib_existing==null) {
            delaunay_array[v][k] = v_neib;
            nn_distance_ditribution[v][k] = dist(Pts,va[v],va[v_neib]); // squared euclidean
            continue :outer;
          }
          if (v_neib_existing.?==v_neib) continue :outer ;
        }
      }
    }
  }

  const avgdist = blk: {
    var ad:f32 = 0;
    var count:u32 = 0;
    for (nn_distance_ditribution) |nnd| {
      for (nnd) |dq| {
        if (dq) |d| {
          ad += d;
          count += 1;
        }
      }
    }
    break :blk ad / @intToFloat(f32,count);
  };



  // print("\n",.{});
  // for (delaunay_array) |da| {
  //   print("{d}\n", .{da});
  // }

  // for each pair of vertices v0,v1 on a delaunay edge we have an associated cost for their translation difference
  // we also have a cost based on the displacement v0(t),v0(t+1)
  // we can pick a vertex at random and choose it's lowest cost match. then given that assignment we can fill in the rest.

  var vertQ = PriorityQueue(TNeibsAssigned, void, gt_TNeibsAssigned).init(allocator, {});
  defer vertQ.deinit();
  try vertQ.add(.{.idx=50 , .nneibs=0});

  // Greedily make assignments for each vertex in the queue based based on minimum strain cost
  // Select vertices by largest number of already-assigned-neighbours 
  while (vertQ.count()>0) {

    // this is the next vertex to match
    const v = vertQ.remove();

    // FIXME allow for divisions
    switch (a_status[v.idx]) {
      .parent => continue,
      .disappear => continue,
      else => {},
    }

    // find best match from among all vb based on strain costs
    var bestcost:?f32  = null;
    var bestidx:?usize = null;
    for (vb) |x_vb,vb_idx| {

      // skip vb if already assigned
      if (b_status[vb_idx] == .daughter) continue;

      const x_va = va[v.idx];
      const nn_cost:f32 = dist(Pts,x_va,x_vb);

      // avoid long jumps
      if (nn_cost > avgdist*2) continue;

      // compute strain cost
      const dx = x_vb - x_va;
      var dx_cost:f32 = 0;
      for (delaunay_array[v.idx]) |va_neib_idx| {  

        const a_idx = if (va_neib_idx) |_v| _v else continue;
        if (a_status[a_idx] != .parent) continue;

        const b_idx = asgn_a2b[a_idx][0].?;
        const dx_va_neib = vb[b_idx] - va[a_idx];
        dx_cost += dist(Pts, dx, dx_va_neib);
      }

      // cost=0 for first vertex in queue (no neibs). then use nearest-neib cost.
      if (dx_cost==0) {
        // add velgrad cost (if any exist)
        if (v.idx==0) {
          print("va_idx={} , vb_idx={}\n",.{v.idx,vb_idx});
          print("bingo dog: {}\n",.{nn_cost});
        }
        dx_cost=nn_cost;
      }

      if (bestcost==null or dx_cost<bestcost.?) {
        bestcost = dx_cost;
        bestidx  = vb_idx;
      }

    }

    if (v.idx==0) {
      // print("va_idx={} , vb_idx={}\n",.{v.idx,vb_idx});
      // print("bingo dog: {}\n",.{nn_cost});
      print("best: {} {}\n", .{bestcost, bestidx});
    }

    // update cell status and graph relations
    if (bestidx) |b_idx| {
      a_status[v.idx] = .parent;
      b_status[b_idx] = .daughter;
      asgn_a2b[v.idx][0] = @intCast(u32,b_idx);
      asgn_b2a[b_idx] = v.idx;
    } else {
      a_status[v.idx] = .disappear;
    }

    // add delaunay neibs of `v` to the PriorityQueue
    for (delaunay_array[v.idx]) |va_neib| {
      if (va_neib==null) continue;
      va_neib_count[va_neib.?] += 1;
      try vertQ.add(.{.idx=va_neib.? , .nneibs=va_neib_count[va_neib.?]});
    }

  }
  

  var hist:[5]u32 = .{0}**5;
  for (a_status) |s| {
    hist[@enumToInt(s)] += 1;
  }

  print("\n",.{});
  print("Assignment Histogram\n", .{});
  for (hist) |h,i| {
    const e = @intToEnum(VStatusTag,i);
    print("{s}→{d}\n", .{@tagName(e), h});
  }


  // print("The Assignments are...\n", .{});
  // for (asgn_a2b) |b_idx,a_idx| {
  //   print("{d}→{d} , {} \n", .{a_idx,b_idx, a_status[a_idx]});
  // }

  return asgn_b2a;
}

test "track. greedy Strain Tracking 2D" {

  print("\n\n",.{});

  const na = 101;
  const nb = 102;

  var va:[na]Pts = undefined;
  for (va) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10 };
  var vb:[nb]Pts = undefined;
  for (vb) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10 };

  const b2a = try strain_track(va[0..],vb[0..]);
  defer allocator.free(b2a);
}




fn argmax1d(comptime T:type, arr:[]T) struct{max:T , idx:usize} {
  var max = arr[0];
  var idx:usize = 0;
  for (arr) |v,i| {
    if (v>max) {
      idx = i;
      max = v;
    }
  }
  return .{.max=max , .idx=idx};
}


// pub fn main() !void {
test "track. greedy min-cost tracking 3D" {

  print("\n\n",.{});

  const na = 1001;
  const nb = 1002;

  var va:[na]Pts3 = undefined;
  for (va) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10, random.float(f32)*10};
  var vb:[nb]Pts3 = undefined;
  for (vb) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10, random.float(f32)*10};

  const parents = try greedy_track(Pts3,va[0..],vb[0..]);
  defer allocator.free(parents);
}

test "track. greedy min-cost tracking 2D" {

  print("\n\n",.{});

  const na = 101;
  const nb = 102;

  var va:[na]Pts = undefined;
  for (va) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10 };
  var vb:[nb]Pts = undefined;
  for (vb) |*v| v.* = .{random.float(f32)*10, random.float(f32)*10 };

  const parents = try greedy_track(Pts,va[0..],vb[0..]);
  defer allocator.free(parents);
}



export fn greedy_track2d(va:[*]f32, na:u32 , vb:[*]f32, nb:u32, res:[*]i32) i32 {

  const va_ = allocator.alloc(Pts,na) catch return -1;
  for (va_) |*v,i| v.* = Pts{va[2*i], va[2*i+1]};
  const vb_ = allocator.alloc(Pts,nb) catch return -1;
  for (vb_) |*v,i| v.* = Pts{vb[2*i], vb[2*i+1]};

  const parents = greedy_track(Pts,va_,vb_) catch return -1;
  defer allocator.free(parents);

  for (parents) |p,i| res[i] = p;

  return 0;
}



pub fn greedy_track(comptime T:type, va:[]T,vb:[]T) ![]i32 {

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
  pairwise_distances(T,cost,va[0..],vb[0..]);

  var asgn = try allocator.alloc(u8,na*nb);
  defer allocator.free(asgn);
  for (asgn) |*v| v.* = 0;


  // sort costs
  // continue adding costs cheapest-first as long as they don't violate asgn constraints
  // sort each row by smallest cost? then 
  var edgeQ = PriorityQueue(CostEdgePair, void, lessThan).init(allocator, {});
  defer edgeQ.deinit();
  for (cost) |c,i| {
    const ia = i / nb;
    const ib = i % nb;
    try edgeQ.add(.{.cost=c,.ia=@intCast(u32,ia),.ib=@intCast(u32,ib)});
  }


  // greedily go through edges and add them to graph iff they don't violate constraints
  var count:usize = 0;
  while (true) {

    count += 1;
    if (count == na*nb + 1) break;

    const edge = edgeQ.remove();
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


const Pts3 = [3]f32;
const Pts  = del.Vec2;


// array order is [a,b]. i.e. a has stride nb. b has stride 1.
fn pairwise_distances(comptime T:type, cost:[]f32, a:[]T, b:[]T) void {
  const na = a.len;
  const nb = b.len;
  assert(cost.len == na*nb);
  for (a) |x,i| {
    for (b) |y,j| {
      cost[i*nb + j] = dist(T,x,y);
    }
  }
}

fn dist(comptime T:type, x:T, y:T) f32 {
  return switch (T){
    Pts  => (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]),
    Pts3 => (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2]),
    else => unreachable,
  };
}

const TNeibsAssigned = struct{idx:u32,nneibs:u8};
fn gt_TNeibsAssigned(context: void, a: TNeibsAssigned, b: TNeibsAssigned) std.math.Order {
  _ = context;
  return std.math.order(a.nneibs, b.nneibs);
}





