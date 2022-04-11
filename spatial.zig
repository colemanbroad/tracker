const std = @import("std");
const print = std.debug.print;
const assert = std.debug.assert;
const expect = std.testing.expect;

const im = @import("imageBase.zig");

var prng = std.rand.DefaultPrng.init(0);
const random = prng.random();
const Allocator = std.mem.Allocator;
var allocator = std.testing.allocator;

const x0 = random.float(f32)*100.0;
const y0 = random.float(f32)*100.0;

const Vec2 = geo.Vec2;
const clipi = geo.clipi;

const Range = geo.Range;

const test_home = @import("tester.zig").test_path ++ "spatial/";
const mkdirIgnoreExistsAbsolute = @import("tester.zig").mkdirIgnoreExistsAbsolute;
test {
  try mkdirIgnoreExistsAbsolute(test_home);
  std.testing.refAllDecls(@This());
}

// test "spatial. bin random points onto 2D grid" {

//   var xs:[100]f32 = undefined;
//   for (xs) |*v| v.* = random.float(f32)*100.0;
//   var ys:[100]f32 = undefined;
//   for (ys) |*v| v.* = random.float(f32)*100.0;

//   var grid = try im.Img2D(u8).init(10,10);
//   defer grid.deinit();

//   for (xs) |x,i| {

//     const y = ys[i];
//     const nx = @floatToInt(usize, x/10);
//     const ny = @floatToInt(usize, y/10);

//     grid.img[nx*10 + ny] += 1;
//   }

//   var rgba = try im.Img2D([4]u8).init(10,10);
//   defer rgba.deinit();

//   for (grid.img) |g,i| {
//     const r = @intCast(u8, (@intCast(u16,g)*10)%255);
//     rgba.img[i] = .{r,r,r,255};
//   }
//   try im.saveRGBA(rgba , test_home ++ "spatial.tga");
// }

test "spatial. GridHash" {

  var pts:[100]Vec2 = undefined;
  for (pts) |*v| v.* = .{random.float(f32)*100.0 , random.float(f32)*100.0};

  var gh = try GridHash.init(allocator,10,10,6,pts[0..],null);
  defer gh.deinit();
  for (pts) |p| {
    _ = gh.neibs(p);
    // print("{}→{d}\n",.{i,gh.neibs(p)}); 
  }

}


const geo = @import("geometry.zig");
const BBox = geo.BBox;

/// Points in 2D are placed into bins on a grid.
/// The grid bins each spatial dimension and we remember a simple affine coordinate transformation for the bounds
/// of the points.
/// A grid with N boxes trained on pts with min=3.2, max=9.8 will divide space into bins of width dx=(9.8-3.2)/N, so
/// any points in [3.2,3.2+dx) are mapped to idx=0, and ...
///
/// dx = (max-min)/nx
/// points in [min + n*dx,min + (n+1)*dx) → grid[n]
/// therefore, a point at x' maps to floor((x'-xmin)/dx)
/// How can we efficiently figure out nx? That's tricky. If we have a radius constraint for NN checks, then it should just be the radius.

const floor = std.math.floor;

const GridHash = struct {

    const Self = @This();
    const Elem = ?u16;
    const SetRoot = u16;
    const IdsDists = struct{ids:[]u16 , dists:[]f32 , n_ids:*usize};

    map: []Elem,
    nx: u32,
    ny: u32,
    // dx: f32,
    // dy: f32,
    nelemax: u32,
    allo:Allocator,
    bb:geo.BBox,
    pts:[]Vec2,



    pub fn init(allo:Allocator,
                nx:u32,
                ny:u32,
                nelemax:u32,
                _pts:[]Vec2,
                labels:?[]u16,
                ) !Self {

      var pts = try allo.alloc(Vec2,_pts.len);
      for (pts) |*p,i| p.* = _pts[i];
      const map = try allocator.alloc(Elem, nx*ny*nelemax);
      // print("\n\nmap.len = {}\n",.{map.len});
      errdefer allocator.free(map);
      for (map) |*v| v.* = null;

      var bb = geo.boundsBBox(pts);
      const ddx = 0.05*(bb.x.hi-bb.x.lo);
      const ddy = 0.05*(bb.y.hi-bb.y.lo);

      bb.x.lo += -ddx;
      bb.x.hi +=  ddx;
      bb.y.lo += -ddy;
      bb.y.hi +=  ddy;

      // const dx = (bb.x.hi-bb.x.lo)/@intToFloat(f32,nx);
      // const dy = (bb.y.hi-bb.y.lo)/@intToFloat(f32,ny);

      outer: for (pts) |p,i| { // i = pt label

        const l = if (labels) |lab| lab[i] else i;

        const ix = x2grid(p[0],bb.x,nx);
        const iy = x2grid(p[1],bb.y,ny);

        const idx = (ix*ny + iy)*nelemax;
        // print("ix,iy,i,idx = {},{},{},{}\n", .{ix,iy,i,idx});

        for (map[idx..idx+nelemax]) |*m| {
          if (m.*==null) {
            m.* = @intCast(u16,l);
            continue :outer ;
          }
        }

        return error.PointDensityError;
      }

      return Self{
                  .allo=allo,
                  .map=map,
                  .nx=nx,
                  .ny=ny,
                  .nelemax=nelemax,
                  // .dx=dx,
                  // .dy=dy,
                  .bb=bb,
                  .pts=pts,
                };
    }

    pub fn deinit(self:Self) void {
      self.allo.free(self.map);
      self.allo.free(self.pts);
    }

    // pub fn pt2grid(p:Vec2,bb:BBox,nx:u16,ny:u16) [2]u16 {
    //   const dx = (bb.x.hi-bb.x.lo)/@intToFloat(f32,nx);
    //   const dy = (bb.y.hi-bb.y.lo)/@intToFloat(f32,ny);
    //   const ix = clipi(u16, @floatToInt(u16,floor((p[0]-bb.x.lo)/dx)) , 0, nx-1);
    //   const iy = clipi(u16, @floatToInt(u16,floor((p[1]-bb.y.lo)/dy)) , 0, ny-1);
    //   return .{ix,iy};
    // }

    pub fn x2grid(x:f32,xr:Range,nx:u32) u32 {
      const dx = (xr.hi-xr.lo)/@intToFloat(f32,nx);
      // print("floor((x-xr.lo)/dx) = {}\n", .{floor((x-xr.lo)/dx)});
      const ix = @floatToInt(i32,floor((x-xr.lo)/dx));
      if (ix<0) return 0;
      if (ix>nx-1) return nx-1;
      return @intCast(u32,ix);
    }


    pub fn neibs(self:Self, p:Vec2) []Elem {
      // const ixiy = pt2grid(p,self.bb,self.nx,self.ny);
      // const ix=ixiy[0];
      // const iy=ixiy[1];
      const ix = x2grid(p[0] , self.bb.x , self.nx);
      const iy = x2grid(p[1] , self.bb.y , self.ny);
      // print("neibs ix iy {} {}\n", .{ix,iy});
      const idx = (ix*self.ny + iy)*self.nelemax;
      return self.map[idx..idx+self.nelemax];
    }

    // first search pairwise in grid, then if need more points expand to surroundings.
    // search all boxes within `radius` of `p`
    pub fn nnRadius(self:Self, res:IdsDists, p:Vec2, radius:f32) !void {

      const ix_min = x2grid(p[0]-radius , self.bb.x , self.nx);
      const ix_max = x2grid(p[0]+radius , self.bb.x , self.nx);
      const iy_min = x2grid(p[1]-radius , self.bb.y , self.ny);
      const iy_max = x2grid(p[1]+radius , self.bb.y , self.ny);

      // const nx = ix_max-ix_min+1;
      // const ny = iy_max-iy_min+1;

      var nn_ids = res.ids;
      var dists  = res.dists;

      var nn_count:usize = 0;
      var xid = ix_min;
      while (xid<=ix_max) : (xid+=1) {
      var yid = iy_min;
      while (yid<=iy_max) : (yid+=1) {
        const idx = (xid*self.ny + yid)*self.nelemax;
        // print("idx {} \n", .{idx});
        const bin = self.map[idx..idx+self.nelemax];
        // print("p {} xid {} yid {} bin {d}\n", .{p,xid,yid,bin});
        // print("neibs {d} \n", .{self.neibs(p)});
        for (bin) |e_| {
          const e = if (e_) |e| e else continue;
          const pt_e = self.pts[e];
          const delta = p-pt_e;
          const dist = @sqrt(@reduce(.Add,delta*delta));

          if (dist<radius) {
            // print("adding e={}\n",.{e});
            nn_ids[nn_count] = e;
            dists[nn_count] = dist;
            nn_count += 1;
          }
        }

      }
      }

      res.n_ids.* = nn_count;

      // remove undefined regions
      // nn_ids = al.shrink(nn_ids,nn_count);
      // dists = al.shrink(dists,nn_count);
      // return IdsDists{.ids=nn_ids , .dists=dists};

    }
};


const pairwise_distances = @import("track.zig").pairwise_distances;

// pub fn main() !void {

test "spatial. radius neibs" {

  const N = 5_000;
  var pts:[N]Vec2 = undefined;
  for (pts) |*v| v.* = .{random.float(f32)*100.0 , random.float(f32)*100.0};
  const sqrtN = @floatToInt(u16,@sqrt(@intToFloat(f32,N)));
  var gh = try GridHash.init(allocator,sqrtN,sqrtN,20,pts[0..],null);
  defer gh.deinit();
  
  const pairdist = try pairwise_distances(allocator,Vec2,pts[0..],pts[0..]);
  defer allocator.free(pairdist);

  // Prealloc idx and dist memory for fast multiple queries
  var ids  = try allocator.alloc(u16,200);
  defer allocator.free(ids);
  var dists = try allocator.alloc(f32,200);
  defer allocator.free(dists);
  var n_ids:usize = 200;
  var res = GridHash.IdsDists{.ids=ids , .dists=dists , .n_ids=&n_ids};

  // Prealloc pdneibs buffer;
  var buf = try allocator.alloc(u16,200);
  defer allocator.free(buf);

  for (pts) |p,i| {
    try gh.nnRadius(res,p,1.0);
    var res_view = res.ids[0..res.n_ids.*]; // leave res.ids length const. Change size of view.


    const pdneibs = blk: {
      var count:u16 = 0;
      for (pts) |_,j| {
        if (pairdist[i*N + j] < 1.0) {
          buf[count] = @intCast(u16,j);
          count += 1;
        }
      }
      break :blk buf[0..count];
    };
  
    // Now compare the results
    std.sort.sort(u16,res_view, {}, comptime std.sort.asc(u16));
    // print("{d}...{d}\n", .{pdneibs,s.ids});
    std.testing.expect(std.mem.eql(u16, pdneibs, res_view)) catch {
      print("\n\nERROR\n\npd={d} , res.ids={d}\n\n",.{pdneibs,res_view});
      unreachable;
    };
  }
}

// pub fn main() !void {
test "spatial. radius speed test" {

  var alltimes:[4][6]i128 = undefined;

  for (alltimes) |*timez| {

  const N = 5_000;
  var pts:[N]Vec2 = undefined;
  for (pts) |*v| v.* = .{random.float(f32)*100.0 , random.float(f32)*100.0};

  var qpts:[N]Vec2 = undefined;
  for (qpts) |*v| v.* = .{random.float(f32)*100.0 , random.float(f32)*100.0};

  const t0 = std.time.nanoTimestamp();

  const sqrtN = @floatToInt(u16,@sqrt(@intToFloat(f32,N)));
  var gh = try GridHash.init(allocator,sqrtN,sqrtN,20,pts[0..],null);
  defer gh.deinit();  

  const t1 = std.time.nanoTimestamp();

  // Prealloc idx and dist memory for fast multiple queries
  var idx  = try allocator.alloc(u16,200);
  defer allocator.free(idx);
  var dist = try allocator.alloc(f32,200);
  defer allocator.free(dist);
  var n_ids:usize = 200;
  var res = GridHash.IdsDists{.ids=idx , .dists=dist , .n_ids=&n_ids};

  const t1a = std.time.nanoTimestamp();

  for (qpts) |q| {
    _ = try gh.nnRadius(res,q,1.0);
  }

  const t2 = std.time.nanoTimestamp();

  // const pairdist = try pairwise_distances(allocator,Vec2,pts[0..],pts[0..]);
  // defer allocator.free(pairdist);

  const t3 = std.time.nanoTimestamp();

  var buf = try allocator.alloc(u16,200);
  defer allocator.free(buf);

  for (qpts) |q| {
    _ = blk: {
      var count:u16 = 0;
      for (pts) |p,j| {
        // if (pairdist[i*N + j] < 1.0) {
        const dx = q-p;
        const pq_dist = @sqrt(@reduce(.Add,dx*dx));
        if (pq_dist < 1.0) {
          // print("pqd = {}   ", .{pq_dist});
          buf[count] = @intCast(u16,j);
          count += 1;
        }
      }
      // buf = allocator.shrink(buf,count);
      break :blk buf;
    };
    // defer allocator.free(neibs);
  }

  const t4 = std.time.nanoTimestamp();
  const times = .{t0-t0,t1-t0,t1a-t0,t2-t0,t3-t0,t4-t0};
  timez.* = times;
  }

  print("\nTiming\n",.{});
  for (alltimes) |_,i|{
    print("\n", .{});
    for (alltimes[0]) |_,j|{
      const t = @intToFloat(f32,alltimes[i][j]) / 1e6;
      print("{d:.3} ",.{t});
    }
  }

  // var mean:[5]f32 = undefined;
  // for (alltimes) |_,i| {
  // for (alltimes[0]) |_,j| {
  //     mean[i] += alltimes[i][j];
  // }
  // }
  // for (mean) |*v,i| v.* /= alltimes[0].len;

  // var stddev:[5]f32 = undefined;
  // for (alltimes) |_,i| {
  // for (alltimes[0]) |_,j| {
  //     const x = alltimes[i][j] - mean[i];
  //     std[i] += x*x;
  // }
  // }

  // stddev[i] 
  // for (stddev) |*s,i| s.* -= mean[i]*mean[i];

  // print("\n\ntimings...{d}\n", .{alltimes});
  // print("\n\nmean...{d}\n", .{mean});
  // print("\n\nstd...{d}\n", .{stddev});
}



