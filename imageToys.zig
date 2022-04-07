const std = @import("std");
const im = @import("imageBase.zig");

const cc = @import("c.zig");
const geo = @import("geometry.zig");
const draw = @import("drawing.zig");
const mesh = @import("mesh.zig");

const Vec2 = geo.Vec2;
const Vec3 = geo.Vec3;

const pi = 3.14159265359;

const print  = std.debug.print;
const assert = std.debug.assert;
const expect = std.testing.expect;
const clamp  = std.math.clamp;

const abs = geo.abs;
const Mat3x3 = geo.Mat3x3;
const Ray = geo.Ray;

const normalize = geo.normalize;
const intersectRayAABB = geo.intersectRayAABB;
const cross = geo.cross;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var allocator = gpa.allocator();
var prng = std.rand.DefaultPrng.init(0);
const rando = prng.random();

const Vector = std.meta.Vector;
const BoxPoly = mesh.BoxPoly;

const Img2D = im.Img2D;
const Img3D = im.Img3D;


// Projections and Cameras and Rendering

/// Orthographic projection along z.
pub fn orthProj(comptime T: type, image: Img3D(T)) ![]T {
  const nz = image.nz;
  const ny = image.ny;
  const nx = image.nx;
  var z: u32 = 0;
  var x: u32 = 0;
  var y: u32 = 0;

  // fixed const for now
  const dim = 0;

  switch (dim) {

    // Poject over Z. Order of dimensions is Z,Y,X. So X is fast and Z is slow.
    0 => {
      var res = try allocator.alloc(T, ny * nx);
      const nxy = nx * ny;

      z = 0;
      while (z < nz) : (z += 1) {
        const z2 = z * nxy;

        y = 0;
        while (y < ny) : (y += 1) {
          const y2 = y * nx;

          x = 0;
          while (x < nx) : (x += 1) {
            if (res[y2 + x] < image.img[z2 + y2 + x]) {
              
              res[y2 + x] = image.img[z2 + y2 + x];
            
            }
          }
        }
      }
      return res;
    },
    else => {
      unreachable;
    },
  }
}

test "imageToys. test orthProj()" {
  var nx: u32 = 200;
  var ny: u32 = 100;
  var nz: u32 = 76;
  const nxy = nx * ny;

  // Initialize memory to 0. Set bresenham global state.
  var img = try allocator.alloc(u8, nz * ny * nx);
  defer allocator.free(img);
  for (img) |*v| v.* = 0;

  // Generate 100 random 3D star shapes. We include boundary conditions here! This is
  // shape dependent. Might be better to separate this out into a separate call to `clamp`.
  {
    var i: u16 = 0;
    while (i < 100) : (i += 1) {
      const x0 = 1 + @intCast(u32, rando.uintLessThan(u32, nx - 2));
      const y0 = 1 + @intCast(u32, rando.uintLessThan(u32, ny - 2));
      const z0 = 1 + @intCast(u32, rando.uintLessThan(u32, nz - 2));

      // Add markers as star
      img[z0 * nxy + y0 * nx + x0] = 255;
      img[z0 * nxy + y0 * nx + x0 - 1] = 255;
      img[z0 * nxy + y0 * nx + x0 + 1] = 255;
      img[z0 * nxy + (y0 - 1) * nx + x0] = 255;
      img[z0 * nxy + (y0 + 1) * nx + x0] = 255;
      img[(z0 - 1) * nxy + y0 * nx + x0] = 255;
      img[(z0 + 1) * nxy + y0 * nx + x0] = 255;
    }
  }

  // Now let's project the image down to 2D and save it.
  var image = Img3D(u8){.img=img,.nx=200,.ny=100,.nz=76};
  const res = try orthProj(u8, image);
  try im.saveU8AsTGAGrey(res, 100, 200, "testOrthProj.tga");
}


/// Max projection with perspective (pure Zig)
/// Uses 👇 `tform()` internal function
pub fn perspectiveProjectionBasic(comptime T: type, image:Img3D(T), nyOut:u32, nxOut:u32, theta:f32 ) ![]f32 {

  const nz = image.nz;
  // const ny = image.ny;
  // const nx = image.nx;

  // transform to get (float) location in input array from output array pixel coordinates
  const tform = struct {
    fn f(image2:Img3D(T), x_out:u32, y_out:u32, z_in:u32, nxOut2:u32, nyOut2:u32, theta_:f32) ?f32 {
      // 1. translate center of output to align with XY center of volume image
      // 2. translate Z to medium, negative value
      // 3. XY (real) = zPer*sin(theta_)

      const _nx_in  = @intToFloat(f32,image2.nx);
      const _ny_in  = @intToFloat(f32,image2.ny);
      const _nz_in  = @intToFloat(f32,image2.nz);
      const _nx     = @intToFloat(f32,nxOut2);
      const _ny     = @intToFloat(f32,nyOut2);
      const fz_in   = @intToFloat(f32,z_in);

      // translate [0..n_out] -> [-1,1] for X,Y,Z
      var fx = (@intToFloat(f32,x_out) - _nx/2) / (_nx/2);
      var fy = (@intToFloat(f32,y_out) - _ny/2) / (_ny/2); 
      var fz = (@intToFloat(f32,z_in) - _nz_in/2) / (_nz_in/2);
      // rotate 
      const fx2 = @cos(theta_)*fx + @sin(theta_)*fz;
      const fz2 = @cos(theta_)*fz - @sin(theta_)*fx;
      fx = fx2;
      fz = fz2;
      // z-dependent rescale
      fx *= 1.1*fz_in/_nz_in * 1.4;
      fy *= 1.1*fz_in/_nz_in * 1.4;
      // translate back to Img3D pixel coords. [-1,1] -> [0..n_in]
      fx += 1;
      fy += 1;
      fz += 1;
      fx *= _nx_in/2;
      fy *= _ny_in/2;
      fz *= _nz_in/2;

      // boundary conds
      if (fx < 0 or fx > _nx_in-1) {return null;}
      if (fy < 0 or fy > _ny_in-1) {return null;}
      if (fz < 0 or fz > _nz_in-1) {return null;}

      // return interp3DUnknown(image2, fz_in, fy, fx);
      return interp3DLinear(image2, fz, fy, fx);
      // return interp3DUnknown(image2, fz_in, fy, fx);
    }
  }.f;

  // Poject over Z. Order of dimensions is Z,Y,X. So X is fast and Z is slow.
  // IMPORTANT: We're dealing with multiple coordinate systems here. Don't confuse them.
  // 1. The YX coordinates of the output array (over which we loop)
  // 2. The ZYX coordinates of the transpormed perspective space (z=[imagePlane, background] y=[-1,1], x=[-1,1]) (we loop over Z).
  // 3. The ZYX real space (orthonormal) coordinates 
  // 4. The XYZ integer coordinates of the input image
    
  var res = try allocator.alloc(f32, nyOut * nxOut);
  
  // coordinates and bounds for the output array   
  var x: u32 = 0;
  var y: u32 = 0;
  var z: u32 = 0;




  z = 0;
  while (z < nz) : (z += 1) 
  {
  
  y = 0;
  while (y < nyOut) : (y += 1) 
  {

  const y2 = y * nxOut;
  
  x = 0;
  while (x < nxOut) : (x += 1) 
  {
          
    // The kernel of the loop. If visited pixel is new Max, then 
    // set it as the result.
    const imgVal = tform(image,x,y,z,nxOut,nyOut,theta);
    if (imgVal) |val| { // skip if null (out of bounds)
      if (res[y2 + x] < val) {
        res[y2 + x] = val;
      }
    }
        
  }
  }
  }

  return res;
}

/// Relies on PerspectiveCamera for coordinate transformations between:
/// 1. 3D world coordinates/// 2. 3D camera view coordinates
/// 3. 2D pixel array coordinates
pub fn perspectiveProjection2(image:Img3D(f32), cam: *PerspectiveCamera) void {


  // Camera and view parameters
  // const camPt = normalize(camPt_);
  const box = Vec3{ @intToFloat(f32,image.nz) , @intToFloat(f32,image.ny) , @intToFloat(f32,image.nx) };
  const boxMidpoint = box / @splat(3,@as(f32,2));
  cam.refocus(cam.loc,boxMidpoint);

  var projmax:f32 = 0;


  // Loop over the projection screen. For each pixel in the screen we cast a ray out along `z` in the screen coordinates,
  // and take the max value along that ray.
  var iy:u32 = 0;
  while (iy < cam.nyPixels) : (iy += 1) {
  const y2 = iy * cam.nxPixels;
  var ix:u32 = 0;

  while (ix < cam.nxPixels) : (ix += 1) {

    // const r0 = cam.transPixel2World(ix,iy,1);  // returns Ray from camera pixel in World Coordinates
    const v0 = cam.pix2world(.{.x=ix,.y=iy});
    // Test for intersection points of Ray with image volume. 
    // Skip this ray if we don't intersect box.
    var intersection = geo.intersectRayAABB( Ray{.pt0=cam.loc , .pt1=v0} , Ray{.pt0=.{0,0,0},.pt1=box} );
    // var intersection = intersectRayAABB( r0 , Ray{.pt0=.{0,0,0} , .pt1=box} ); // returns Ray with 2 intersection points
    const v1 = normalize(v0 - cam.loc);

    // skip this Ray if it doesn't intersect with box
    if (intersection.pt0==null and intersection.pt1==null) continue;
    if (intersection.pt0==null) intersection.pt0 = cam.loc; // equal to r0.pt0;
    if (intersection.pt1==null) intersection.pt1 = cam.loc; // equal to r0.pt0;
    // Now that both points are well defined we can cast rays from inside the volume!

    // Take maximum value along the intersecting line segment.
    const intersectionLength = abs(intersection.pt1.? - intersection.pt0.?);
    var iz_f:f32 = 3;
    while (iz_f < intersectionLength-1) : (iz_f += 1)
    {

      // NOTE: We start at intersection.pt0 always, because we define it to be nearest to starting point.
      // v0 is unit vec pointing away from camera
      const v4 = intersection.pt0.? + v1*@splat(3,iz_f);
      
      // print("p0 = {d}\n",.{intersection.pt0.?});
      // print("v1 = {d}\n",.{v1});
      // print("v4 = {d}\n",.{v4});

      // if @reduce(.Or, v4<0 or v4>box-1) continue; // TODO: The compiler will probably allow this one day.
      // if (@reduce(.Or, v4<@splat(3,@as(f32,0))) or @reduce(.Or,v4>box - @splat(3,@as(f32,1)))) continue; // TODO: The compiler will probably allow this one day.

      if (v4[0] < 0 or v4[0] > box[0]-1) continue; //{break :blk null;}
      if (v4[1] < 0 or v4[1] > box[1]-1) continue; //{break :blk null;}
      if (v4[2] < 0 or v4[2] > box[2]-1) continue; //{break :blk null;}

      const val = interp3DLinear(image, v4[0], v4[1], v4[2]);

      if (cam.screen[y2 + ix] < val) {
        cam.screen[y2 + ix] = val;
        if (val > projmax) projmax = val;
      }
      
  }}}

  // Add bounding box consisting of 8 vertices connected by 12 lines separating 6 faces
  // the 12 lines come from the following. We have a low box corner [0,0,0] and a high corner [nx,ny,nz].
  // If we start with choosing the low value in each of the x,y,z dimensions we can get the next three lines 
  // by choosing the high value in exactly one of the three dims, i.e. [0,0,0] is connected to [1,0,0],[0,1,0],[0,0,1] which each sum to 1
  // and [1,0,0] is connected to [0,0,0] and [1,1,0],[1,0,1] which sum to 2. 
  // The pattern we are describing is a [Hasse Diagram](https://mathworld.wolfram.com/HasseDiagram.html)

  // var nx:u32=cam.nxPixels;
  // var ny:u32=cam.nyPixels;
  // cc.bres.init(&cam.screen[0],&ny,&nx);
  const container = Img2D(f32){.img=cam.screen , .nx=cam.nxPixels , .ny=cam.nyPixels};
  const poly = BoxPoly.createAABB(.{0,0,0}, box);
  for (poly.es) |e| {
    const p0 = cam.world2pix(poly.vs[e[0]]);
    const p1 = cam.world2pix(poly.vs[e[1]]);
    // plotLine(f32,container,p0.x,p0.y,p1.x,p1.y,1.0);
    // cc.bres.plotLine(@intCast(i32,p0.x) , @intCast(i32,p0.y) , @intCast(i32,p1.x) , @intCast(i32,p1.y));
    const x0 = @intCast(i32,p0.x);
    const y0 = @intCast(i32,p0.y);
    const x1 = @intCast(i32,p1.x);
    const y1 = @intCast(i32,p1.y);
    draw.drawLineInBounds(f32,container,x0, y0, x1, y1, projmax);
  }

  // DONE
}

test "imageXXXToys. render stars with perspectiveProjection2()"{
  // pub fn main() !void {

  try mkdirIgnoreExists("renderStarsWPerspective");
  print("\n",.{});

  var img = try randomStars();
  defer allocator.free(img.img); // FIXME

  var nameStr = try std.fmt.allocPrint(allocator, "renderStarsWPerspective/img{:0>4}.tga", .{0});

  const traj = sphereTrajectory();
  
  // {var i:u32=0; while(i<5):(i+=1){
  {var i:u32=0; while(i<traj.len):(i+=1){

    // const i_ = @intToFloat(f32,i);
    const camPt = traj[i]*Vec3{900,900,900};
    // print("\n{d}",.{camPt});  
    // const camPt = Vec3{400,50,50};
    var cam2 = try PerspectiveCamera.init(camPt, .{0,0,0}, 401, 301, null,);

    perspectiveProjection2(img,&cam2);

    nameStr = try std.fmt.bufPrint(nameStr, "renderStarsWPerspective/img{:0>4}.tga", .{i});
    print("{s}\n", .{nameStr});
    try im.saveF32AsTGAGreyNormed(cam2.screen, 301, 401, nameStr);

  }}

  // var nameStr = try std.fmt.allocPrint(allocator, "rotproj/projImagePointsPerspective{:0>4}.tga", .{0}); // filename
  // try im.saveU8AsTGAGrey(allocator, res, 100, 200, "projImagePoints.tga");
}



/// Max projection with perspective (pure Zig)
/// Poject over Z. Order of dimensions is Z,Y,X. So X is fast and Z is slow.
/// IMPORTANT: We're dealing with multiple coordinate systems here. Don't confuse them.
/// 1. The YX coordinates of the output array (over which we loop)
/// 2. The ZYX coordinates of the transpormed perspective space (z=[imagePlane, background] y=[-1,1], x=[-1,1]) (we loop over Z).
/// 3. The ZYX real space (orthonormal) coordinates 
/// 4. The XYZ integer coordinates of the input image
pub fn perspectiveProjection(comptime T: type, image:Img3D(T), nyOut:u32, nxOut:u32, camPt_:Vec3 ) ![]f32 {

  // print("\nminmax: {d}\n", .{im.minmax(f32, image.img)});

  // Camera and view parameters
  const camPt = geo.normalize(camPt_);
  const nz = image.nz;
  const ny = image.ny;
  const nx = image.nx;
  const box = Vec3{ @intToFloat(f32,nz) , @intToFloat(f32,ny) , @intToFloat(f32,nx) };
  const boxMidpoint = box / @splat(3,@as(f32,2));
  const cameraWC = boxMidpoint + camPt_ * @splat(3,abs(box));  // WC = World Coordinates
  const nyOut_f = @intToFloat(f32,nyOut);
  const nxOut_f = @intToFloat(f32,nxOut);
  const aspectRatio = Vec3{1 , 0.1 , 0.1 * nxOut_f / nyOut_f};
  const rotMat = cameraRotation(camPt, .ZYX); // specify axis order
  // print("RotMat = {d}\n",.{rotMat});

  // // Get corners of camera screen in world coordinates
  // const screen = .{
  //   .topRight = matVecMul(rotMat, Vec3{-1, 0.05, 0.1}),
  //   .botRight = matVecMul(rotMat, Vec3{-1,-0.05, 0.1}),
  //   .topLeft  = matVecMul(rotMat, Vec3{-1, 0.05,-0.1}),
  //   .botLeft  = matVecMul(rotMat, Vec3{-1,-0.05,-0.1}),
  // };
  // print("Screen in World Coordinates: {d}\n",.{screen});


  // Store result of projection in `res`
  var res = try allocator.alloc(f32, nyOut * nxOut);
  for (res) |*v| v.* = 0;

  // Loop over the projection screen. For each pixel in the screen we cast a ray out along `z` in the screen coordinates,
  // and take the max value along that ray.
  var iy:u32 = 0;
  while (iy < nyOut) : (iy += 1) {
  const y2 = iy * nxOut;
  var ix:u32 = 0;
  while (ix < nxOut) : (ix += 1) {

    // Convert intToFloat for ease of use
    const iy_f = @intToFloat(f32,iy);
    const ix_f = @intToFloat(f32,ix);
    // points from origin to screen pixel in normalized coordinatess
    const v0 = Vec3{1 , iy_f/nyOut_f - 0.5 , ix_f/nxOut_f - 0.5} * aspectRatio;
    
    // Rotate v0 s.t. it aligns with the camera view. 
    // This little vector is used to step through the volume
    // TODO: why does this seem to work with our ZYX coordinates?
    // const v1 = rotateCwithRotorAB(v0, z0, -camPt);
    const v1 = geo.matVecMul(rotMat,v0);

    // // `start` lives on the camera's screen in world coordinates.
    // const start = blk: {
    //   const v1_centered = v1 + camPt;
    //   const v1_centered_scaled = v1_centered * @splat(3, abs(box)/2);
    //   const v1_centered_scaled_translated = v1_centered_scaled + cameraWC ;
    //   break :blk v1_centered_scaled_translated;
    // };

    // Test for intersection points of Ray with image volume. 
    // Skip this ray if we don't intersect box.
    // const intersection = intersectRayAABB( Ray{.pt0=start,.pt1=start-v1} , Ray{.pt0=.{0,0,0},.pt1=box} );
    const intersection = intersectRayAABB( Ray{.pt0=cameraWC,.pt1=cameraWC-v1} , Ray{.pt0=.{0,0,0},.pt1=box} );

    // print("{d} \n",.{intersection});

    // skip this Ray if it doesn't intersect with box
    if (intersection.pt0==null or intersection.pt1==null) continue;

    // If we intersect the box for some distance then take maxium along the intersecting line segment.
    const intersectionLength = abs(intersection.pt1.? - intersection.pt0.?);
    var iz_f:f32 = 0;
    while (iz_f < intersectionLength) : (iz_f += 1)
    {

      // NOTE: We start at intersection.pt0 always, because we define it to be nearest to starting point.
      // v1 points away from origin, so we must subtract it.
      const v4 = intersection.pt0.? - v1*@splat(3,iz_f);

      // if (iy==ny/2 and ix==nx/2) {
      //   print("start = {d}\n", .{start});
      //   print("v4 = {d}\n", .{v4});
      //   print("box = {d}\n", .{box});
      // }

      const imgVal = blk: {
        // boundary conditions
        // TODO: we may be able to avoid this check, as we have already performed the Ray-Box intersection test.
        if (v4[0] < 0 or v4[0] > @intToFloat(f32, nz-1)) {break :blk null;}
        if (v4[1] < 0 or v4[1] > @intToFloat(f32, ny-1)) {break :blk null;}
        if (v4[2] < 0 or v4[2] > @intToFloat(f32, nx-1)) {break :blk null;}
        // otherwise LERP
        break :blk interp3DLinear(image, v4[0], v4[1], v4[2]);
      };

      // skip if null (ray is out of bounds)
      if (imgVal) |val| { 
        if (res[y2 + ix] < val) {
          res[y2 + ix] = val;
        }
      }
      
  }}}


  // OK, Now that we've finished casting rays from the camera through the volume let's add the bounding box lines.
  // This involves the reverse projection, from the bounding box back onto the camera screen.
  // We should be able to compute the screen coordinates of our bounding box by first finding the world coordinates of our screen!
  // Then we just need to find the intersection between our world object -> camera ray and the screen rectangle...

    // // Convert intToFloat for ease of use
    // const iy_f = @intToFloat(f32,iy);
    // const ix_f = @intToFloat(f32,ix);
    // // points from origin to screen pixel in normalized coordinatess
    // const v0 = Vec3{1 , iy_f/nyOut_f - 0.5 , ix_f/nxOut_f - 0.5} * aspectRatio;
    
    // // Rotate v0 s.t. it aligns with the camera view. 
    // // This little vector is used to step through the volume
    // // TODO: why does this seem to work with our ZYX coordinates?
    // // const v1 = rotateCwithRotorAB(v0, z0, -camPt);
    // const v1 = matVecMul(rotMat,v0);

    // // now find starting point.
    // const start = blk: {
    //   const v1_centered = v1 + camPt;
    //   const v1_centered_scaled = v1_centered * @splat(3, abs(box)/2);
    //   const v1_centered_scaled_translated = v1_centered_scaled + cameraWC ;
    //   break :blk v1_centered_scaled_translated;
    // };

  return res;
}

test "imageToys. render stars with perspectiveProjection()" {

  try mkdirIgnoreExists("renderStarsWPerspectiveV1");

  print("\n",.{});

  // Build a 2x2x2 image with the pixel values 1..8 
  var img = try randomStars();
  defer allocator.free(img.img); // FIXME
  var nameStr = try std.fmt.allocPrint(allocator, "renderStarsWPerspectiveV1/img{:0>4}.tga", .{0}); // filename

  const traj = sphereTrajectory();
  
  {var i:u32=0; while(i<10):(i+=1){

    const i_ = @intToFloat(f32,i);
    const camPt = traj[i*10]*@splat(3, i_*4/10 + 1);

    // const res = try perspectiveProjectionBasic(f32, img, 200, 400, 2*3.14159*i_/20);
    const res = try perspectiveProjection(f32, img, 200, 400, camPt);
    nameStr = try std.fmt.bufPrint(nameStr, "renderStarsWPerspectiveV1/img{:0>4}.tga", .{i});
    print("{s}\n", .{nameStr});
    try im.saveF32AsTGAGreyNormed(res, 200, 400, nameStr);
    }}

  // var nameStr = try std.fmt.allocPrint(allocator, "rotproj/projImagePointsPerspective{:0>4}.tga", .{0}); // filename
  // try im.saveU8AsTGAGrey(allocator, res, 100, 200, "projImagePoints.tga");
}


const AxisOrder = enum {XYZ,ZYX};

// Construct an orthogonal rotation matrix which aligns z->camera and y->z
// `camPt_` uses 👇 normalized coordinates and points toward the origin [0,0,0].
fn cameraRotation(camPt_:Vec3,axisOrder:AxisOrder) Mat3x3 {

  // standardize on XYZ axis order
  const camPt = switch (axisOrder) {
    .XYZ => camPt_,
    .ZYX => Vec3{camPt_[2],camPt_[1],camPt_[0]},
  };

  const x1 = Vec3{1,0,0};
  // const y1 = Vec3{0,1,0};
  const z1 = Vec3{0,0,1};
  const z2 = geo.normalize(camPt);
  const x2 = if (@reduce(.And,z1==z2)) x1 else normalize(cross(z1,z2)); // protect against z1==z2.
  const y2 = normalize(cross(z2,x2));

  const rotM = geo.matFromVecs(x2,y2,z2);

  // return in specified axis order
  switch (axisOrder) {
    .XYZ => return rotM ,
    .ZYX => return Mat3x3{rotM[8],rotM[7],rotM[6],rotM[5],rotM[4],rotM[3],rotM[2],rotM[1],rotM[0],} ,
  }
}

test "imageToys. cameraRotation()" {
  // fn testCameraRotation() !void {
  // begin with XYZ coords, then swap to ZYX
  const x1 = Vec3{1,0,0};
  const y1 = Vec3{0,1,0};
  const z1 = Vec3{0,0,1};

  const camPt = Vec3{-1,-1,-1}; // checks
  const z2 = geo.normalize(camPt);
  const x2 = geo.normalize(geo.cross(z1,z2));
  const y2 = geo.normalize(geo.cross(z2,x2));

  const rotM = geo.matFromVecs(x2,y2,z2);

  print("\n",.{});
  print("Rotated x1: {d} \n", .{geo.matVecMul(rotM,x1)});
  print("Rotated y1: {d} \n", .{geo.matVecMul(rotM,y1)});
  print("Rotated z1: {d} \n", .{geo.matVecMul(rotM,z1)});

  try expect(x2[2]==0);
  try expect(abs(cross(x2,y2)-z2) < 1e-6);
}
const SCREENX:u16 = 2880;
const SCREENY:u16 = 1800;

// NOTE: By convention, the camera faces the -z direction, which allows y=UP, x=RIGHT in a right handed coordinate system.
const PerspectiveTransform = struct {
  loc : Vec3,          // location of camera in world coordinates
  pointOfFocus : Vec3, // point of focus in world coordinates (no simulated focal plane, so any ray is equivalent for now.)
  // nxPixels : u32,      // Assume square pixels. nx/ny defines the aspect ratio (aperture)
  // nyPixels : u32,      // Assume square pixels. nx/ny defines the aspect ratio (aperture)
  apertureX : f32,          // Field of view in the horizontal (x) direction. Units s.t. fov=1 produces 62° view angle. nx/ny_pixels determines FOV in y (vertical direction).
  apertureY : f32,
  axes : Mat3x3,       // orthonormal axes of camera coordinate system (in world coordinates)
  axesInv : Mat3x3,    
  // screen : []f32,      // where picture data is recorded
  // screenFace : [4]Vec3, // four points which define aperture polygon (in world coordinates)

  pub fn init(loc : Vec3,
              pointOfFocus : Vec3,
              // nxPixels : u32,     
              // nyPixels : u32,     
              _apertureX : ?f32,
              _apertureY : ?f32,
              ) !@This() {

    var apertureX = if (_apertureX) |fx| fx else 0.2;
    var apertureY = if (_apertureY) |fy| fy else 0.2*@intToFloat(f32,SCREENY)/@intToFloat(f32,SCREENX);
    // var apertureY = apertureX*@intToFloat(f32,nyPixels)/@intToFloat(f32,nxPixels);
    // var screen = try allocator.alloc(f32,nxPixels*nyPixels);
    var axes = cameraRotation(loc - pointOfFocus, .ZYX);
    for (axes) |v| assert(v!=std.math.nan_f32);
    var axesInv = geo.invert3x3(axes);
    for (axesInv) |v| assert(v!=std.math.nan_f32);

    var this = @This(){
      .loc=loc,
      .pointOfFocus=pointOfFocus,
      // .nxPixels=nxPixels,
      // .nyPixels=nyPixels,
      .apertureX=apertureX,
      .apertureY=apertureY,
      .axes=axes,
      .axesInv=axesInv,
      // .screen=screen,
      // .screenFace=undefined,
    };

    // // update screenFace
    // var sf0 = this.pix2world(0,0);
    // var sf1 = this.pix2world(nxPixels,0);
    // var sf2 = this.pix2world(nxPixels,nyPixels);
    // var sf3 = this.pix2world(0,nyPixels);
    // this.screenFace = .{sf0,sf1,sf2,sf2};

    return this;
  }

  // pub fn deinit(this:@This()) void {
  //   allocator.free(this.screen);
  // }

  // world2cam(cam.loc) = {0,0,0}
  // world2cam(cam.pointOfFocus) = {dist,0,0} 
  pub fn world2cam(this:@This(), v0:Vec3) Vec3 {
    const v1 = v0 - this.loc;
    const v2 = geo.matVecMul(this.axesInv,v1);
    return v2;
  }

  // cam2world({0,0,0}) = cam.loc
  // cam2world(worldOrigin) = {0,0,0}
  // cam2world(focalPoint) = cam.pointOfFocus
  pub fn cam2world(this:@This(), v0:Vec3) Vec3 {
    // const p0 = Vec3{-abs(this.loc), 0 , 0}; // location of world origin in camera coordinates
    const v1 = geo.matVecMul(this.axes, v0);
    const v2 = v1 + this.loc;
    return v2;
    // const v1 = v0 + this.loc; // translate origin to [0,0,0]
    // const v2 = matVecMul(this.axes, v1); // rotate 
  }

  // const Px = struct{x:i64,y:i64};

  // // returns [x,y] pixel coordinates
  // pub fn world2pix(this:@This(), v0:Vec3) Px {
  //   const v1 = this.world2cam(v0);
  //   const v2 = v1 / @splat(3,-v1[0]); // divide by -Z to normalize to -1 (homogeneous coords)
  //   const ny = @intToFloat(f32,this.nyPixels);
  //   const nx = @intToFloat(f32,this.nxPixels);
  //   const v3 = (v2 - Vec3{0,-this.apertureY/2,-this.apertureX/2}) / Vec3{1,this.apertureY,this.apertureX} * Vec3{1,ny,nx};
  //   // const v3 = (v2 + Vec3{0,this.apertureY/2,this.apertureX/2}) * Vec3{1,ny,nx} / Vec3{1,this.apertureY,this.apertureX};
  //   const v4 = @floor(v3);
  //   const y  = @floatToInt(i64,v4[1]);
  //   const x  = @floatToInt(i64,v4[2]);
  //   return .{.x=x,.y=y};
  // }

  // // a pixel (input) collects light from all points along a Ray (return)
  // // NOTE: px are allowed to be i64 (we can refer to px outside the image boundaries)
  // pub fn pix2world(this:@This(), px : Px ) Vec3 {
  //   const _x = (@intToFloat(f32,px.x) / @intToFloat(f32,this.nxPixels-1) - 0.5) * this.apertureX; // map pixel values onto [-0.5,0.5] inclusive
  //   const _y = (@intToFloat(f32,px.y) / @intToFloat(f32,this.nyPixels-1) - 0.5) * this.apertureY; // map pixel values onto [-0.5,0.5] inclusive
  //   const v0 = Vec3{-1,_y,_x};
  //   const v1 = this.cam2world(v0);
  //   // return Ray{.pt0=this.loc, .pt1=v1};
  //   return v1;
  // }

  // pub fn refocus(this: *@This() , loc:Vec3 , pointOfFocus:Vec3) void {
  //   this.axes = cameraRotation(loc - pointOfFocus, .ZYX);
  //   this.axesInv = invert3x3(this.axes); // TODO: this should just be a transpose
  //   this.loc = loc;
  //   this.pointOfFocus = pointOfFocus;
  // }
};

// returns new vertices which have been projected onto 2D with perspective view from camera with relative position `camPosition`.
pub fn rotate2cam(a:std.mem.Allocator , pts:[]Vec3 , camPosition:Vec3) ![]Vec3 {
  const verts = try a.alloc(Vec3 , pts.len);

  const bds = geo.bounds3(pts);
  const width = bds[1]-bds[0];
  const midpoint = (bds[1] + bds[0]) / Vec3{2,2,2};
  // const spin = sphereTrajectory();
  const dist2focus = abs(width) * 10; // x aperture is 0.2 by default. abs(width*5) should be just enough to fit all points in view.
  const campt = geo.normalize(camPosition);
  const rvec = Vec3{dist2focus,dist2focus,dist2focus};

  const cam = try PerspectiveTransform.init(campt*rvec , .{0,0,0} , null , null); // focus at origin

  for (pts) |p,i| {
    // const pcam = cam.world2cam(p - midpoint);
    // verts[i] = .{pcam[2],pcam[1]}; // X,Y 
    verts[i] = cam.world2cam(p - midpoint);
  }

  return verts;
}

// NOTE: By convention, the camera faces the -z direction, which allows y=UP, x=RIGHT in a right handed coordinate system.
pub const PerspectiveCamera = struct {
  loc : Vec3,          // location of camera in world coordinates
  pointOfFocus : Vec3, // point of focus in world coordinates (no simulated focal plane, so any ray is equivalent for now.)
  nxPixels : u32,      // Assume square pixels. nx/ny defines the aspect ratio (aperture)
  nyPixels : u32,      // Assume square pixels. nx/ny defines the aspect ratio (aperture)
  apertureX : f32,          // Field of view in the horizontal (x) direction. Units s.t. fov=1 produces 62° view angle. nx/ny_pixels determines FOV in y (vertical direction).
  apertureY : f32,
  axes : Mat3x3,       // orthonormal axes of camera coordinate system (in world coordinates)
  axesInv : Mat3x3,    
  screen : []f32,      // where picture data is recorded
  // screenFace : [4]Vec3, // four points which define aperture polygon (in world coordinates)

  pub fn init(loc : Vec3,
              pointOfFocus : Vec3,
              nxPixels : u32,     
              nyPixels : u32,     
              _apertureX : ?f32,
              ) !@This() {

    var apertureX = if (_apertureX) |fx| fx else 0.2;
    var apertureY = apertureX*@intToFloat(f32,nyPixels)/@intToFloat(f32,nxPixels);
    var screen = try allocator.alloc(f32,nxPixels*nyPixels);
    var axes = cameraRotation(loc - pointOfFocus, .ZYX);
    for (axes) |v| assert(v!=std.math.nan_f32);
    var axesInv = geo.invert3x3(axes);
    for (axesInv) |v| assert(v!=std.math.nan_f32);

    var this = @This(){
      .loc=loc,
      .pointOfFocus=pointOfFocus,
      .nxPixels=nxPixels,
      .nyPixels=nyPixels,
      .apertureX=apertureX,
      .apertureY=apertureY,
      .axes=axes,
      .axesInv=axesInv,
      .screen=screen,
      // .screenFace=undefined,
    };

    // // update screenFace
    // var sf0 = this.pix2world(0,0);
    // var sf1 = this.pix2world(nxPixels,0);
    // var sf2 = this.pix2world(nxPixels,nyPixels);
    // var sf3 = this.pix2world(0,nyPixels);
    // this.screenFace = .{sf0,sf1,sf2,sf2};

    return this;
  }

  pub fn deinit(this:@This()) void {
    allocator.free(this.screen);
  }

  // world2cam(cam.loc) = {0,0,0}
  // world2cam(cam.pointOfFocus) = {dist,0,0} 
  pub fn world2cam(this:@This(), v0:Vec3) Vec3 {
    const v1 = v0 - this.loc;
    const v2 = geo.matVecMul(this.axesInv,v1);
    return v2;
  }

  // cam2world({0,0,0}) = cam.loc
  // cam2world(worldOrigin) = {0,0,0}
  // cam2world(focalPoint) = cam.pointOfFocus
  pub fn cam2world(this:@This(), v0:Vec3) Vec3 {
    // const p0 = Vec3{-abs(this.loc), 0 , 0}; // location of world origin in camera coordinates
    const v1 = geo.matVecMul(this.axes, v0);
    const v2 = v1 + this.loc;
    return v2;
    // const v1 = v0 + this.loc; // translate origin to [0,0,0]
    // const v2 = matVecMul(this.axes, v1); // rotate 
  }

  const Px = struct{x:i64,y:i64};

  // returns [x,y] pixel coordinates
  pub fn world2pix(this:@This(), v0:Vec3) Px {
    const v1 = this.world2cam(v0);
    const v2 = v1 / @splat(3,-v1[0]); // divide by -Z to normalize to -1 (homogeneous coords)
    const ny = @intToFloat(f32,this.nyPixels);
    const nx = @intToFloat(f32,this.nxPixels);
    const v3 = (v2 - Vec3{0,-this.apertureY/2,-this.apertureX/2}) / Vec3{1,this.apertureY,this.apertureX} * Vec3{1,ny,nx};
    // const v3 = (v2 + Vec3{0,this.apertureY/2,this.apertureX/2}) * Vec3{1,ny,nx} / Vec3{1,this.apertureY,this.apertureX};
    const v4 = @floor(v3);
    const y  = @floatToInt(i64,v4[1]);
    const x  = @floatToInt(i64,v4[2]);
    return .{.x=x,.y=y};
  }

  // a pixel (input) collects light from all points along a Ray (return)
  // NOTE: px are allowed to be i64 (we can refer to px outside the image boundaries)
  pub fn pix2world(this:@This(), px : Px ) Vec3 {
    const _x = (@intToFloat(f32,px.x) / @intToFloat(f32,this.nxPixels-1) - 0.5) * this.apertureX; // map pixel values onto [-0.5,0.5] inclusive
    const _y = (@intToFloat(f32,px.y) / @intToFloat(f32,this.nyPixels-1) - 0.5) * this.apertureY; // map pixel values onto [-0.5,0.5] inclusive
    const v0 = Vec3{-1,_y,_x};
    const v1 = this.cam2world(v0);
    // return Ray{.pt0=this.loc, .pt1=v1};
    return v1;
  }

  pub fn refocus(this: *@This() , loc:Vec3 , pointOfFocus:Vec3) void {
    this.axes = cameraRotation(loc - pointOfFocus, .ZYX);
    this.axesInv = geo.invert3x3(this.axes); // TODO: this should just be a transpose
    this.loc = loc;
    this.pointOfFocus = pointOfFocus;
  }
};

test "imageToys. test all PerspectiveCamera transformations" {
  //   try camTest();
  // }
  // pub fn camTest() !void {
  var cam = try PerspectiveCamera.init(.{100,100,100}, .{50,50,50}, 401, 301, null, );
  print("\n",.{});

  try expect( @reduce(.And, cam.world2cam(cam.loc) == Vec3{0,0,0}) );
  print("cam.loc in cam coordinates : {d:.3} \n", .{cam.world2cam(cam.loc)});
  const p1 = cam.world2cam(cam.pointOfFocus);
  print("cam.pointOfFocus in cam coordinates : {d:.3} \n", .{p1});
  try expect(p1[1]==0); try expect(p1[2]==0);
  const p0 = cam.world2cam(.{0,0,0}); // origin in cam coordinates
  print("origin in cam coordinates : {d:.3} \n", .{p0});
  print("origin back to world coordinates : {d:.3} \n", .{cam.cam2world(p0)});
  print("pointOfFocus : {d:.3} \n", .{cam.cam2world(cam.world2cam(cam.pointOfFocus))});

  print("\n\nFuzz Testing\n\n", .{});
  var i:u32=0;
  while (i<10):(i+=1) {
    const p2 = geo.randNormalVec3();
    // const z  = p2[0];
    // print("p2 = {d}\n",.{p2});

    const d0 = cam.cam2world(cam.world2cam(p2)) - p2;
    // print("c2w.w2c ... d0 = {d}\n",.{d0});
    try expect(abs(d0) < 1e-4);

    const d2 = cam.world2cam(cam.cam2world(p2)) - p2;
    // print("c2w.w2c ... d2 = {d}\n",.{d2});
    try expect(abs(d2) < 1e-4);


    const rpx = .{.x=rando.int(u8) , .y=rando.int(u8)};
    const pt2 = cam.pix2world(rpx);
    const pt3 = (pt2-cam.loc)*Vec3{1,1,1} + cam.loc;
    const px2 = cam.world2pix(pt3);
    // print("rpx : {d}\n", .{rpx});
    print("px2 : {d}\n", .{px2});



    // const px0 = cam.world2pix(p2);
    // print("px0 = {d}\n",.{px0});
    // const pt3 = cam.pix2world(px0);
    // print("pt3 = {d}\n",.{pt3});

    // const r1 = Ray{.pt0=cam.loc , .pt1=pt3};
    // const d1 = closestApproachRayPt(r1,p2);
    // print("w2p.p2w ... d1 = {d}\n",.{d1});
    // try expect(abs(d1-p2) < 1e-4);


  }
}

test "imageToys. Rotating Spiral w world2pix()" {
  // pub fn main() !void {

  try mkdirIgnoreExists("rotatingSpiral");

  const pts = sphereTrajectory();

  var nameStr = try allocator.alloc(u8,40);

  var k:u32 = 0;
  while (k<100):(k+=1){

    const fp = pts[k] * Vec3{300,300,300};
    var cam = try PerspectiveCamera.init(fp,
                  .{0,0,0},
                  401, // need an odd number of pixel s.t. x,y=0,0 maps to pixel 200,150.
                  301, // need an odd number of pixel s.t. x,y=0,0 maps to pixel 200,150.
                  null,
                  );

    const nx = cam.nxPixels;
    const ny = cam.nyPixels;

    for (pts) |v0| {
      const v1 = v0 * Vec3{15,15,15};
      const px = cam.world2pix(v1);
      if (0 <= px.x and px.x < nx and 0 <= px.y and px.y < ny){
        cam.screen[@intCast(u32, px.y * nx + px.x)] = 1.0;
      }
    }

    nameStr = try std.fmt.bufPrint(nameStr, "rotatingSpiral/img{:0>4}.tga", .{k});
    // var nameStr = "camW2S/img000.tga";
    // print("{s}\n", .{nameStr});
    try im.saveF32AsTGAGreyNormed(cam.screen, 301, 401, nameStr);
}}








// computes derivative from current state of lorenz system
// NOTE: sig=10 rho=28 beta=8/3 give cool manifold
fn lorenzEquation(xyz:Vec3 , sig:f32 , rho:f32 , beta:f32 ) Vec3 {
  const x = xyz[0];
  const y = xyz[1];
  const z = xyz[2];

  const dx = sig*(y-x);
  const dy = x*(rho-z) - y;
  const dz = x*y - beta*z;

  return Vec3{dx,dy,dz};
}

// for building small, stack-sized arrays returned by value.
// TODO: Is there a way to use this same function for producing runtime arrays?
// Or, is there a way of evaluating the runtime version of this function at comptime and getting
// an array instead of a slice ? 
fn nparange(comptime T:type , comptime low:T , comptime hi:T , comptime inc:T) blk: {
    const n = @floatToInt(u16,(hi-low)/inc + 1);
    break :blk [n]T;
}
  {
  const n = @floatToInt(u16,(hi-low)/inc + 1);
  var arr:[n]T = undefined;
  for (arr) |*v,i| v.* = low + @intToFloat(f32,i)*inc;
  return arr;
}






// SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE
// SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE
// SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE   SAVE AN IMAGE



// in place affine transform to place verts inside (nx,ny) box with 5% margins
pub fn fitbox(verts:[][2]f32 , nx:u32 , ny:u32) void {

  const xborder = 0.05 * @intToFloat(f32,nx);
  const xwidth  = @intToFloat(f32,nx) - 2*xborder;
  const yborder = 0.05 * @intToFloat(f32,ny);
  const ywidth  = @intToFloat(f32,ny) - 2*yborder;

  // const mima = minmax(verts);
  const mima = im.bounds2(verts);
  const mi = mima[0];
  const ma = mima[1];

  const xrenorm = xwidth / (ma[0]-mi[0]);
  const yrenorm = ywidth / (ma[1]-mi[1]);

  for (verts) |*v| {
    const x = (v.*[0]-mi[0]) * xrenorm + xborder;
    const y = (v.*[1]-mi[1]) * yrenorm + yborder;
    v.* = .{x,y};
  }
}

pub fn sum(comptime n: u8, T:type, vals:[][n]T) [n]T {
  var res = [1]T{0}**n;
  comptime var count=0;

  for (vals) |v| {
    inline while (count<n):(count+=1) {
      res[count] += v[count];
    }
  }
  return res;
}

pub fn fitboxiso(verts:[][2]f32 , nx:u32 , ny:u32) void {

  const xborder = 0.05 * @intToFloat(f32,nx);
  const xwidth  = @intToFloat(f32,nx) - 2*xborder;
  const yborder = 0.05 * @intToFloat(f32,ny);
  const ywidth  = @intToFloat(f32,ny) - 2*yborder;

  const mima = im.bounds2(verts);
  const mi = mima[0];
  const ma = mima[1];
  const xrenorm = xwidth / (ma[0]-mi[0]);
  const yrenorm = ywidth / (ma[1]-mi[1]);

  const renorm = min(xrenorm,yrenorm);
  var mean = sum(2,f32,verts);
  mean[0] /= @intToFloat(f32, verts.len );
  mean[1] /= @intToFloat(f32, verts.len );

  for (verts) |*v| {
    const x = (v.*[0]-mean[0]) * renorm + @intToFloat(f32,nx)/2;
    const y = (v.*[1]-mean[1]) * renorm + @intToFloat(f32,ny)/2;
    v.* = .{x,y};
  }
}




// IMAGE FILTERS  IMAGE FILTERS  IMAGE FILTERS  IMAGE FILTERS  IMAGE FILTERS  IMAGE FILTERS
// IMAGE FILTERS  IMAGE FILTERS  IMAGE FILTERS  IMAGE FILTERS  IMAGE FILTERS  IMAGE FILTERS
// IMAGE FILTERS  IMAGE FILTERS  IMAGE FILTERS  IMAGE FILTERS  IMAGE FILTERS  IMAGE FILTERS

// XY format . TODO: ensure inline ?
pub inline fn inbounds(img:anytype , px:anytype) bool {
  if (0 <= px[0] and px[0]<img.nx and 0 <= px[1] and px[1]<img.ny) return true else return false;
}

// Run a simple min-kernel over the image to remove noise.
fn minfilter(img:Img2D(f32)) !void {
  const nx = img.nx;
  // const ny = img.ny;
  const s = img.img; // source
  const t = try allocator.alloc(f32,s.len); // target
  const deltas = [_]Vector(2,i32){ .{-1,0} , .{0,1} , .{1,0} , .{0,-1} ,.{0,0}};

  for (s) |_,i| {
    // const i = @intCast(u32,_i);
    var mn = s[i];
    const px = Vector(2,i32){ @intCast(i32,i%nx) , @intCast(i32,i/nx) };
    for (deltas) |dpx| {
      const p = px + dpx;
      const v = if (inbounds(img,p)) s[@intCast(u32,p[0]) + nx*@intCast(u32,p[1])] else 0;
      mn = min(mn , v);
    }
    t[i] = mn;
  }

  // for (s) |_,i| {
  // }
  for (img.img) |*v,i| {
    v.* = t[i];
  }
}

// Run a simple min-kernel over the image to remove noise.
fn blurfilter(img:Img2D(f32)) !void {
  const nx = img.nx;
  // const ny = img.ny;
  const s = img.img; // source
  const t = try allocator.alloc(f32,s.len); // target
  const deltas = [_]Vector(2,i32){ .{-1,0} , .{0,1} , .{1,0} , .{0,-1} ,.{0,0}};

  for (s) |_,i| {
    // const i = @intCast(u32,_i);
    var x = @as(f32,0); //s[i];
    const px = Vector(2,i32){ @intCast(i32,i%nx) , @intCast(i32,i/nx) };
    for (deltas) |dpx| {
      const p = px + dpx;
      const v = if (inbounds(img,p)) s[@intCast(u32,p[0]) + nx*@intCast(u32,p[1])] else 0;
      x += v;
    }
    t[i] = x/5;
  }

  // for (s) |_,i| {
  // }
  for (img.img) |*v,i| {
    v.* = t[i];
  }
}






test "imageToys. bitsets" {
  // pub fn main() !void {

  var bitset = std.StaticBitSet(3000).initEmpty();
  bitset.set(0);
  bitset.set(2999);

  var it = bitset.iterator(.{});
  var n = it.next();
  while (n!=null) {
    print("item: {d}\n" , .{n});
    n = it.next();
  }
}



const Str = []const u8;

pub fn mkdirIgnoreExists(dirname:Str) !void {
  std.fs.cwd().makeDir(dirname) catch |e| switch (e) {
    error.PathAlreadyExists => {},
    else => return e ,
  };
}


// to create a stream plot we need to integrate a bunch of points through the vector field.
// note... the form of the "perlinnoise" function is just a 3D scalar field, but if we want a 
// 2D random vector field we just sample with two z values > 2 apart (uncorrelated). The integration
// of the points can be exactly as simple as with Lorenz integrator. But now we need multiple points, and shorter trajectories.
// 
// 
// 1. draw field's vectors 
// 2. draw streamlines
// 3. movie of moving streamlines
// 
// 1. use a grid initial pts
// 2. use random initial pts
// 3. use circle-pack initial pts
// 4. use grid + random deviation
test "imageToys. various stream plots" {
  // pub fn main() !void {
  var nx:u32 = 1200; //3*800/2;
  var ny:u32 = 1200; //3*800/2;
  const pic = try Img2D([4]u8).init(nx,ny);
  const pts = try allocator.alloc(Vec2,50*50);
  defer allocator.free(pts);

  try mkdirIgnoreExists("streamplot");

  const dx = @as(f32,0.1);
  const dy = @as(f32,0.1);

  for (pts) |_,i| {
    const x = @intToFloat(f32, i%50) * dx;
    const y = @intToFloat(f32, i/50) * dy;
    pts[i] = Vec2{x,y};
  }


  // Perlin Noise Vectors located at `pts`
  const vals = try allocator.alloc(Vec2,pts.len);
  defer allocator.free(vals);
  for (vals) |_,i| {
    const p = pts[i];
    const x = ImprovedPerlinNoise.noise(p[0],p[1],10.6);
    const y = ImprovedPerlinNoise.noise(p[0],p[1],14.4); // separated in z by 2 => uncorrelated
    vals[i] = .{@floatCast(f32,x) , @floatCast(f32,y)};
  }

  // const vals = gridnoise();
  for (pic.img) |*v| v.* = .{0,0,0,255};



  // Draw lines on `pic`
  // Let's rescale p and v so they fit on pic.img.
  // Note we can do an isotropic rescaling without changing the properties we care about.
  // currently pts spacing is `dx`, so we should divide by dx to get a spacing of 1 pixel.
  // The bounds are (0,0) and (5,5). So we could also rescale to fit the bounds to the image.
  // for (pts) |_,j| {

  //   const p = pts[j];
  //   const v = vals[j];
  //   const x  = @floatToInt(u31, @intToFloat(f32,ny-50)*p[0]/5 + 25 + @intToFloat(f32,nx-ny)/2);
  //   const y  = @floatToInt(u31, @intToFloat(f32,ny-50)*p[1]/5 + 25);
  //   const x2 = @floatToInt(u31, @intToFloat(f32,x) + 20*v[0] );
  //   const y2 = @floatToInt(u31, @intToFloat(f32,y) + 20*v[1] );

  //   drawLine([4]u8, pic , x , y , x2 , y2 , .{255,255,255,255});
    
  //   // add circle at base
  //   pic.img[x-1 + nx*y]     = .{255/2 , 255/2 , 255/2 , 255};
  //   pic.img[x+1 + nx*y]     = .{255/2 , 255/2 , 255/2 , 255};
  //   pic.img[x + nx*(y-1)]   = .{255/2 , 255/2 , 255/2 , 255};
  //   pic.img[x + nx*(y+1)]   = .{255/2 , 255/2 , 255/2 , 255};
  //   pic.img[x-1 + nx*(y-1)] = .{255/2 , 255/2 , 255/2 , 255};
  //   pic.img[x+1 + nx*(y+1)] = .{255/2 , 255/2 , 255/2 , 255};
  //   pic.img[x+1 + nx*(y-1)] = .{255/2 , 255/2 , 255/2 , 255};
  //   pic.img[x-1 + nx*(y+1)] = .{255/2 , 255/2 , 255/2 , 255};

  // }
  // try im.saveRGBA(pic,"stream1.tga"); // simple vector plot. large base.



  var name = try allocator.alloc(u8,40);
  var time:[200]f32 = undefined;
  // for (time) |*v,i| v.* = @intToFloat(f32,i) / 100;
  const dt = Vec2{0.002,0.002};

  var ptscopy = try allocator.alloc(Vec2 , pts.len);
  for (ptscopy) |*p,i| p.* = pts[i];
  var nextpts = try allocator.alloc(Vec2 , pts.len);
  for (nextpts) |*p,i| p.* = pts[i];

  for (time) |_,j| {
    for (pts) |_,i| {
      var pt = &ptscopy[i];
      var nextpt = &nextpts[i];

      // print("j,i = {d},{d}\n", .{j,i});
      // if (j==1 and i==6) @breakpoint();


      pt.* = nextpt.*;
      // update position
      if (pt.*[0]<0 or pt.*[1]<0) continue;
      const deltax = 10*@floatCast(f32, ImprovedPerlinNoise.noise(pt.*[0],pt.*[1],10.6) );
      const deltay = 10*@floatCast(f32, ImprovedPerlinNoise.noise(pt.*[0],pt.*[1],14.4) ); // separated in z by 2 => uncorrelated
      // print("delta = {d},{d}\n",.{deltax,deltay});
      nextpt.* += Vec2{deltax,deltay} * dt;

      // draw centered & isotropic with borders
      // const x  = @floatToInt(u31, @intToFloat(f32,ny-50)*pt.*[0]/5 + 25 + @intToFloat(f32,nx-ny)/2);
      // const y  = @floatToInt(u31, @intToFloat(f32,ny-50)*pt.*[1]/5 + 25);

      // fit to image (maybe anisotropic)
      const x1  = @floatToInt(u31, @intToFloat(f32,nx)*pt.*[0]/5);
      const y1  = @floatToInt(u31, @intToFloat(f32,ny)*pt.*[1]/5);
      const x2  = @floatToInt(u31, @intToFloat(f32,nx)*nextpt.*[0]/5);
      const y2  = @floatToInt(u31, @intToFloat(f32,ny)*nextpt.*[1]/5);


      if (!inbounds(pic, .{x1,y1}) or !inbounds(pic, .{x2,y2})) continue; 

      const v0 = RGBA.fromBGRAu8(pic.img[x1 + nx*y1]);
      const v1 = RGBA{.r=1 , .g=1 , .b=1 , .a=0.05};
      const v2 = v1.mix2(v0).toBGRAu8();
      draw.drawLine([4]u8 , pic, x1 , y1 , x2 , y2 , v2);
    }

    name = try std.fmt.bufPrint(name, "streamplot/img{:0>4}.tga", .{j});
    try im.saveRGBA(pic,name);
  }

  // var time:u32 = 0;
  // while (time<100) : (time+=1) {
  //   for (pts) |_,i| {
  //     const i_ = @intToFloat(f32,i);
  //   }
  // }
}


/// Everything you want to know about [color blending](https://www.w3.org/TR/compositing-1/#blending) from the WWWC.
pub const RGBA = packed struct {
  r:f32,
  g:f32,
  b:f32,
  a:f32,

  const This = @This();

  // mix low-alpha x (fg) into high-alpha y (bg)
  pub fn mix(fg:This , bg:This) This {
    const alpha = 1 - (1-fg.a) * (1-bg.a);
    if (alpha<1e-6) return .{.r=0 , .g=0 , .b=0 , .a=alpha};
    const w0 = fg.a / alpha;
    const w1 = bg.a * (1 - fg.a) / alpha;
    const red   = fg.r * w0 + bg.r * w1;
    const green = fg.g * w0 + bg.g * w1;
    const blue  = fg.b * w0 + bg.b * w1;
    return .{.r=red , .g=green , .b=blue , .a=alpha};
  }

  pub fn mix2(x:This , y:This) This {
    const alpha = 1 - (1-x.a) * (1-y.a);
    if (alpha<1e-6) return .{.r=0 , .g=0 , .b=0 , .a=alpha};
    const w0 = x.a / (x.a + y.a);
    const w1 = y.a / (x.a + y.a);
    const red   = x.r * w0 + y.r * w1;
    const green = x.g * w0 + y.g * w1;
    const blue  = x.b * w0 + y.b * w1;
    return .{.r=red , .g=green , .b=blue , .a=alpha};
  }

  pub fn fromBGRAu8(a:[4]u8) This {
    const v0 = @intToFloat(f32 , a[0]) / 255;
    const v1 = @intToFloat(f32 , a[1]) / 255;
    const v2 = @intToFloat(f32 , a[2]) / 255;
    const v3 = @intToFloat(f32 , a[3]) / 255;
    return .{.r=v2 , .g=v1 , .b=v0 , .a=v3};
  }

  pub fn toBGRAu8(a:This) [4]u8 {
    const v0 = @floatToInt(u8 , a.b*255);
    const v1 = @floatToInt(u8 , a.g*255);
    const v2 = @floatToInt(u8 , a.r*255);
    const v3 = @floatToInt(u8 , a.a*255);
    return [4]u8{v0,v1,v2,v3};
  }
};

const Mesh = mesh.Mesh;

test "imageToys. render soccerball with occlusion" {
// pub fn main() !void {
  var box = BoxPoly.createAABB(.{3,3,3} , .{85,83,84});
  var a = box.vs.len; var b = box.es.len; var c = box.fs.len; // TODO: FIXME: I shouldn't have to do this just to convert types....
  const surf1 = Mesh{.vs=box.vs[0..a] , .es=box.es[0..b] , .fs=box.fs[0..c]};
  const surf2 = try mesh.subdivideMesh(surf1 , 3);
  defer surf2.deinit();

  try draw.drawMesh3DMovie2(surf2 , "soccerball/img");
}

test "imageToys. random scattering of points with drawPoints2D()" {
  var points2D:[100_000]f32 = undefined; // 100k f32's living on the Stack 
  for (points2D) |*v| v.* = rando.float(f32);
  try draw.drawPoints2D(f32, points2D[0..], "scatter.tga", false);
}

test "imageToys. drawPoints2D() Spiral" {
  // pub fn main() !void {
  const N = 10_000;
  var points2D:[N]f32 = undefined;
  // Arrange points in spiral
  for (points2D) |*v,i| v.* = blk: { 
    const _i = @intToFloat(f32,i);
    if (i%2==0) {
      const x = @cos(30 * _i/N * 6.28)*0.5*_i/N + 0.5;
      break :blk x;
    } else {
      const y = @sin(30 * _i/N * 6.28)*0.5*_i/N + 0.5;
      break :blk y;
    }
  };
  try draw.drawPoints2D(f32, points2D[0..], "spiral.tga", true);
}












// DATA GENERATION  DATA GENERATION
// DATA GENERATION  DATA GENERATION
// DATA GENERATION  DATA GENERATION





pub fn random2DMesh(nx:f32,ny:f32) !struct{verts:[]Vec2 , edges:std.ArrayListUnmanaged([2]u32)} {
  const verts = try allocator.alloc(Vec2,100);
  var edges = try std.ArrayListUnmanaged([2]u32).initCapacity(allocator, 1000); // NOTE: we must use `var` for edges here, even though it's referring to slices on the heap? how?

  for (verts) |*v,i| {
    const x = rando.float(f32) * nx;
    const y = rando.float(f32) * ny;
    v.* = .{x,y};

    const n1 = rando.uintLessThan(u32, @intCast(u32,verts.len));
    const n2 = rando.uintLessThan(u32, @intCast(u32,verts.len));
    const n3 = rando.uintLessThan(u32, @intCast(u32,verts.len));
    edges.appendAssumeCapacity(.{@intCast(u32,i),n1});
    edges.appendAssumeCapacity(.{@intCast(u32,i),n2});
    edges.appendAssumeCapacity(.{@intCast(u32,i),n3});
  }
  const ret = .{.verts=verts, .edges=edges};
  return ret;
}

test "imageToys. random mesh" {
  const pic = try Img2D([4]u8).init(800,800);
  var msh = try random2DMesh(800,800);

  for (msh.edges.items) |e| {
    const x0 = @floatToInt(u31, msh.verts[e[0]][0] );
    const y0 = @floatToInt(u31, msh.verts[e[0]][1] );
    const x1 = @floatToInt(u31, msh.verts[e[1]][0] );
    const y1 = @floatToInt(u31, msh.verts[e[1]][1] );
    draw.drawLine([4]u8,pic,x0, y0, x1, y1,.{255,0,255,255});
  }
  try im.saveRGBA(pic,"meshme.tga");
}

// img volume filled with stars. img shape is 50x100x200 .ZYX
// WARNING: must free() Img3D.img field 
pub fn randomStars() !Img3D(f32) {

  var img = blo:
  {
    var data  = try allocator.alloc(f32, 50*100*200);
    for (data) |*v| v.* = 0;
    var img3d = Img3D(f32) {
        .img = data,
        .nz=50, .ny=100, .nx=200,
        };
    break :blo img3d;
  };

  // Generate 100 random 3D points. We include boundary conditions here! This is
  // shape dependent. Might be better to separate this out into a separate call to `clamp`.
  const nx = img.nx;
  const ny = img.ny;
  const nz = img.nz;
  const nxy = nx*ny;

  var i: u16 = 0;
  while (i < 100) : (i += 1) {
    const x0 = 1 + @intCast(u32, rando.uintLessThan(u32, nx - 2));
    const y0 = 1 + @intCast(u32, rando.uintLessThan(u32, ny - 2));
    const z0 = 1 + @intCast(u32, rando.uintLessThan(u32, nz - 2));

    // Add markers as star
    img.img[z0 * nxy + y0 * nx + x0] = 1.0;
    img.img[z0 * nxy + y0 * nx + x0 - 1] = 1.0;
    img.img[z0 * nxy + y0 * nx + x0 + 1] = 1.0;
    img.img[z0 * nxy + (y0 - 1) * nx + x0] = 1.0;
    img.img[z0 * nxy + (y0 + 1) * nx + x0] = 1.0;
    img.img[(z0 - 1) * nxy + y0 * nx + x0] = 1.0;
    img.img[(z0 + 1) * nxy + y0 * nx + x0] = 1.0;
  }

  return img;
}

// spiral walk around the unit sphere
pub fn sphereTrajectory() [100]Vec3 {
  var phis:[100]f32 = undefined;
  // for (phis) |*v,i| v.* = ((@intToFloat(f32,i)+1)/105) * pi;
  for (phis) |*v,i| v.* = ((@intToFloat(f32,i))/99) * pi;
  var thetas:[100]f32 = undefined;
  // for (thetas) |*v,i| v.* = ((@intToFloat(f32,i)+1)/105) * 2*pi;
  for (thetas) |*v,i| v.* = ((@intToFloat(f32,i))/99) * 2*pi;
  var pts:[100]Vec3 = undefined;
  for (pts) |*v,i| v.* = Vec3{@cos(phis[i]) , @sin(thetas[i])*@sin(phis[i]) , @cos(thetas[i])*@sin(phis[i]) }; // ZYX coords
  return pts;
}

test "imageToys. Perlin noise (improved)" {
  // pub fn main() !void {

  var noise = try allocator.alloc(f32,512*512);
  for (noise) |*v,i| {
    const x = @intToFloat(f64, i/512) / 2;
    const y = @intToFloat(f64, i%512) / 2;
    const z = 0;
    v.* = @floatCast(f32, ImprovedPerlinNoise.noise(x,y,z) );
  }

  try im.saveF32AsTGAGreyNormed(noise, 512, 512, "perlinnoise.tga");
}


// Improved Perlin Noise
// Ported directly from Ken Perlin's Java implementation: https://mrl.cs.nyu.edu/~perlin/noise/
// Correlations fall off to zero at distance=2 ?
const ImprovedPerlinNoise = struct {

   pub fn noise(_x:f64, _y:f64, _z:f64) f64 {
      var x = _x; // @floor(x);                                // FIND RELATIVE X,Y,Z
      var y = _y; // @floor(y);                                // OF POINT IN CUBE.
      var z = _z; // @floor(z);

      var X = @floatToInt(u32, @floor(x)) & @as(u32,255); // & 255;                   // FIND UNIT CUBE THAT
      var Y = @floatToInt(u32, @floor(y)) & @as(u32,255); // & 255;                   // CONTAINS POINT.
      var Z = @floatToInt(u32, @floor(z)) & @as(u32,255); // & 255;
      x -= @floor(x);                                // FIND RELATIVE X,Y,Z
      y -= @floor(y);                                // OF POINT IN CUBE.
      z -= @floor(z);
      var u:f64 = fade(x);                           // COMPUTE FADE CURVES
      var v:f64 = fade(y);                           // FOR EACH OF X,Y,Z.
      var w:f64 = fade(z);
      // HASH COORDINATES OF THE 8 CUBE CORNERS
      var A:u32  = p[X  ]+Y;
      var AA:u32 = p[A]+Z;
      var AB:u32 = p[A+1]+Z;      
      var B:u32  = p[X+1]+Y;
      var BA:u32 = p[B]+Z;
      var BB:u32 = p[B+1]+Z;

      return lerp(w, lerp(v, lerp(u, grad(p[AA  ], x  , y  , z   ),  // AND ADD
                                     grad(p[BA  ], x-1, y  , z   )), // BLENDED
                             lerp(u, grad(p[AB  ], x  , y-1, z   ),  // RESULTS
                                     grad(p[BB  ], x-1, y-1, z   ))),// FROM  8
                     lerp(v, lerp(u, grad(p[AA+1], x  , y  , z-1 ),  // CORNERS
                                     grad(p[BA+1], x-1, y  , z-1 )), // OF CUBE
                             lerp(u, grad(p[AB+1], x  , y-1, z-1 ),
                                     grad(p[BB+1], x-1, y-1, z-1 ))));
   }
   pub fn fade(t:f64) f64 { return t * t * t * (t * (t * 6 - 15) + 10); }
   pub fn lerp(t:f64, a:f64, b:f64) f64 { return a + t * (b - a); }
   pub fn grad(hash:u32, x:f64, y:f64, z:f64) f64 {
      var h:u32 = hash & 15;                      // CONVERT LO 4 BITS OF HASH CODE
      var u:f64 = if (h<8) x else y;                 // INTO 12 GRADIENT DIRECTIONS.
      var v:f64 = if (h<4) y else if (h==12 or h==14) x else z;
      // return (if ((h&1) == 0) u else -u) + ((h&2) == 0 ? v : -v);
      return (if ((h&1)==0) u else -u) + (if ((h&2)==0) v else -v);

   
   }

   const permutation = [_]u32{ 151,160,137,91,90,15,
   131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
   190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
   88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
   77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
   102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
   135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
   5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
   223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
   129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
   251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
   49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
   138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
   };

   // for (int i=0; i < 256 ; i++) p[256+i] = p[i] = permutation[i];
   const p = blk: {
     var _p:[2*permutation.len]u32 = undefined;
     var i:u32=0; 
     while (i<256) : (i+=1) {
         _p[i] = permutation[i];
         _p[256+i] = permutation[i];
     }
     break :blk _p;
   };
};

pub fn gridnoise(nx:u32 , ny:u32) !Img2D(Vec2) {
  const noise = try Img2D(Vec2).init(nx,ny);
  for (noise.img) |*v,i| {
    v.* = Vec2{@intToFloat(f32,i%nx) , @intToFloat(f32,i/nx)};
    v.* += Vec2{rando.float(f32) , rando.float(f32)};
  }
  return noise;
}

test "imageToys. vec syntax" {
  const a = [2]f32{1,9};
  // _ = [2]f32{4,7};
  _ = geo.vec2(a);

  const d:u32 = 100;
  const e:u32 = 200;
  const f = @intCast(i33,d)-e;
  print("\nf={}\n" , .{f});
}





// Various attempts at classical "Linear Interpolation" (LERP)
// NOTE: LERP isn't actually linear!

// multiplies weights. nonstandard.
inline fn interp3DWeird(image3D:Img3D(f32), z:f32, y:f32, x:f32) f32 {
  const nxy = image3D.nx * image3D.ny;
  const nx  = image3D.nx;
  const img = image3D.img;

  // Get indices of six neighbouring hyperplanes

  // const z0 = @floatToInt(u32, @floor(z) );
  // const z1 = @floatToInt(u32, @ceil(z) );
  // const y0 = @floatToInt(u32, @floor(y) );
  // const y1 = @floatToInt(u32, @ceil(y) );
  // const x0 = @floatToInt(u32, @floor(x) );
  // const x1 = @floatToInt(u32, @ceil(x) );

  // // delta vec
  // const dz = @floor(z) - z;
  // const dy = @floor(y) - y;
  // const dx = @floor(x) - x;

  // How do we generalize linear interpolation to nD while maintaining
  // the property that the value is exact at locations of indices?
  // If weights decay linearly and _smoothly_ this may be impossible.
  // A smooth, linear radial decay produces a cone of weights, which cannot be zero at all three
  // neighboring vertices (on a single pixel). One solution is to maintain the 
  // continuity, but to lose continuity of the first derivative!
  // Instead we interpolate a perfectly linear weight function plane that intersects 
  // only _three_ of the four vertices! i.e. the weights go to zero on the diagonal instead
  // of on the opposite vertex.

  // But doesn't this mean we have all zero-valued weights at the center of the voxel?
  // This doesn't work either....

  // OK, another idea. 
  // Each weight function (in 2D) is defined by three points: main vertex, a neighbouring vertex, and the pixel center, with weights (1,0,1/4)!
  // This ensures continuity of the weight function (but not derivative), while allowing the function to be true interpolation, and also ensuring that 
  // the center of a pixel is interpolated as the average of it's four corners.
  // This procedure generalizes to 3D by adding an additional neighbour vertex at which we also fit with weight zero, giving (1,0,0,1/8) as the interpolant vals.

  // A linear interp that decays to 1/8 at the midpoint doesn't work.
  // const r3by2 = 0.8660254037844386; // sqrt 3 / 2 = length of vec(.5, .5, .5)

  // const ws:[8]f32 = undefined;
  // const fs:[8]f32 = undefined;
  var rval:f32 = 0;

  {var i:u8 = 0;
  while (i<8):(i+=1) {
      // vec with 000 as origin DOT vec towards midpoint (direction of fastest decay)
      const vz = @floatToInt(u32, if ((i>>2)%2==0) @floor(z) else @ceil(z) );
      const vy = @floatToInt(u32, if ((i>>1)%2==0) @floor(y) else @ceil(y) );
      const vx = @floatToInt(u32, if ((i>>0)%2==0) @floor(x) else @ceil(x) );

      const dz = if ((i>>2)%2==0) z else 1-z ; // flip axis orientation for each vertex
      const dy = if ((i>>1)%2==0) y else 1-y ; // flip axis orientation for each vertex
      const dx = if ((i>>0)%2==0) x else 1-x ; // flip axis orientation for each vertex

      // We want the dot product to start at 1 (it does) and to decrease linearly in the direction of the midpoint
      // until it reaches a low value there of 1/4.
      const dotval = (dz + dy + dx) / 1.50; // dot product (but with vector to midpoint removed, and then made covariant s.t. dot is 1 at midpoint.)
      // print("dx={d:.2} dy={d:.2} dz={d:.2} ... dot={d:.2}\n", .{dx,dy,dz,dot});
      // const w   = if (dot<1) 1-dot*3/4 else 0;
      const w = @fabs(1-dotval*7/8); //
      // 3/4 gives us 1/4 weight remaining at midpoint. at any point inside the pixel we have 4/8 vertices with nonzero weights.

      const val = img[vz*nxy + vy*nx + vx];
      // print("w={}, val={} \n", .{w,val});
      rval += w * val;
      // ws[i] = w;
      // fs[i] = img[vz*nxy + vy*nx + vx];
  }}

  return rval;
}

// Classical lerp
inline fn interp3DLinear(image3D:Img3D(f32), z:f32, y:f32, x:f32) f32 {
  const nxy = image3D.nx * image3D.ny;
  const nx  = image3D.nx;
  const img = image3D.img;

  // Get indices of six neighbouring hyperplanes
  const z0 = @floatToInt(u32, @floor(z) );
  const z1 = @floatToInt(u32, @ceil(z) );
  const y0 = @floatToInt(u32, @floor(y) );
  const y1 = @floatToInt(u32, @ceil(y) );
  const x0 = @floatToInt(u32, @floor(x) );
  const x1 = @floatToInt(u32, @ceil(x) );

  // delta vec
  const dz = z - @floor(z);
  const dy = y - @floor(y);
  const dx = x - @floor(x);

  // Get values of eight neighbouring pixels
  const f000 = img[z0*nxy + y0*nx + x0];
  const f001 = img[z0*nxy + y0*nx + x1];
  const f010 = img[z0*nxy + y1*nx + x0];
  const f011 = img[z0*nxy + y1*nx + x1];
  const f100 = img[z1*nxy + y0*nx + x0];
  const f101 = img[z1*nxy + y0*nx + x1];
  const f110 = img[z1*nxy + y1*nx + x0];
  const f111 = img[z1*nxy + y1*nx + x1];



  // ┌───┐              ┌───┐                          
  // │x10│              │x11│                          
  // └───┘▲───────────▶▲└───┘                          
  //      │            │                               
  //      │            │                               
  //      │            │      example edge     ┌──────┐
  //      │            │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│x01_11│
  //      │            │                       └──────┘
  //      │            │                               
  //      │───────────▶│                               
  // ┌───┐            ┌───┐                            
  // │x00│            │x01│                            
  // └───┘            └───┘                            


  // Linear interp along Z
  const f00 = dz*f100 + (1-dz)*f000;
  const f01 = dz*f101 + (1-dz)*f001;
  const f10 = dz*f110 + (1-dz)*f010;
  const f11 = dz*f111 + (1-dz)*f011;

  // Linearly interp along Y
  const f01_11 = (1-dy)*f01 + dy*f11;
  const f00_10 = (1-dy)*f00 + dy*f10;

  // Linear interp along X
  const result = (1-dx)*f00_10 + dx*f01_11;

  // const f00_01 = (1-dx)*f00 + dx*f01;
  // const f10_11 = (1-dx)*f10 + dx*f11;
  // Linearly interp along Y
  // const f_x1 = (1-dy)*f00_01 + dy*f10_11;
  // const f_x2 = (1-dy)*f00_01 + dy*f10_11;
  // const f_y_final = (1-dx)*f00_10 + dx*f01_11;
  // const result = (f_x_final + f_y_final) / 2;

  return result;
}

// funny attempt at True Linear lerp
inline fn interp3DUnknown(image3D:Img3D(f32), z:f32,y:f32,x:f32) f32 {
  const nxy = image3D.nx * image3D.ny;
  const nx  = image3D.nx;
  const img = image3D.img;

  // Get indices of six neighbouring hyperplanes
  const z0 = @floatToInt(u32, @floor(z) );
  const z1 = @floatToInt(u32, @ceil(z) );
  const y0 = @floatToInt(u32, @floor(y) );
  const y1 = @floatToInt(u32, @ceil(y) );
  const x0 = @floatToInt(u32, @floor(x) );
  const x1 = @floatToInt(u32, @ceil(x) );

  // Get values of eight neighbouring pixels
  const f000 = img[z0*nxy + y0*nx + x0];
  const f001 = img[z0*nxy + y0*nx + x1];
  const f010 = img[z0*nxy + y1*nx + x0];
  const f011 = img[z0*nxy + y1*nx + x1];
  const f100 = img[z1*nxy + y0*nx + x0];
  const f101 = img[z1*nxy + y0*nx + x1];
  const f110 = img[z1*nxy + y1*nx + x0];
  const f111 = img[z1*nxy + y1*nx + x1];

  // Compute weights for each face
  const wz1 = z - @intToFloat(f32, z0);
  const wy1 = y - @intToFloat(f32, y0);
  const wx1 = x - @intToFloat(f32, x0);
  const wz0 = 1 - wz1;
  const wy0 = 1 - wy1;
  const wx0 = 1 - wx1;

  // Compute weights for each of eight neighbouring pixels
  var w000 = (wz0 * wy0 * wx0);
  var w001 = (wz0 * wy0 * wx1);
  var w010 = (wz0 * wy1 * wx0);
  var w011 = (wz0 * wy1 * wx1);
  var w100 = (wz1 * wy0 * wx0);
  var w101 = (wz1 * wy0 * wx1);
  var w110 = (wz1 * wy1 * wx0);
  var w111 = (wz1 * wy1 * wx1);
  const wtot = (w000 + w001 + w010 + w011 + w100 + w101 + w110 + w111);
  w000 /= wtot ;
  w001 /= wtot ;
  w010 /= wtot ;
  w011 /= wtot ;
  w100 /= wtot ;
  w101 /= wtot ;
  w110 /= wtot ;
  w111 /= wtot ;

  const res = 
      (w000 * f000) +
      (w001 * f001) +
      (w010 * f010) +
      (w011 * f011) +
      (w100 * f100) +
      (w101 * f101) +
      (w110 * f110) +
      (w111 * f111);

  return res;
}


test "imageToys. test trilinear interp"{
  print("\n", .{});

  // Build a 2x2x2 image with the pixel values 1..8 
  const img = blo:
  {
    var _a  = [_]f32{1,2,3,4,5,6,7,8,};
    // var _a  = [_]f32{5,5,5,5,5,5,5,5,};
    var _b = Img3D(f32) {
        .img = &_a,
        .nz=2, .ny=2, .nx=2,
        };
    break :blo _b;
  };

  // Assert that the middle interpolated value is the average across all 8 pixels.
  {
    var r1 = interp3DUnknown(img,0.5,0.5,0.5);
    print("interp3DUnknown Midpoint Value = {}\n", .{r1});
    try std.testing.expect((r1>1) and (r1<8));

    r1 = interp3DWeird(img,0.5,0.5,0.5);
    print("interp3DWeird Midpoint Value = {}\n", .{r1});
    try std.testing.expect(r1==4.5);

    r1 = interp3DLinear(img,0.5,0.5,0.5);
    print("interp3DLinear Midpoint Value = {}\n", .{r1});
    try std.testing.expect((r1>1) and (r1<8));
  }

  // "Interpolate" by taking only the extreme values which must be equal to the pixel values.

  {
    var result:[8]f32 = undefined;
    var i:u4 = 0; // WARNING: if you try to define i:u3 then we get an integer overflow!
    while (i<8):(i+=1)
    {
      var _z = if ((i>>2)%2==0) @as(f32,0) else @as(f32,1);
      var _y = if ((i>>1)%2==0) @as(f32,0) else @as(f32,1);
      var _x = if ((i>>0)%2==0) @as(f32,0) else @as(f32,1);

      const r1 = interp3DUnknown(img,_z,_y,_x); // PASS
      // const r1 = interp3DLinear(img,_z,_y,_x); // PASS
      // const r1 = interp3DWeird(img,_z,_y,_x); // FAIL
      print("{d},{d},{d} --> {d}\n", .{_z,_y,_x, r1});
      result[i] = r1;
    }
    try std.testing.expect(std.mem.eql(f32,&result,img.img));
  }

  print("\n", .{});
}


const max = std.math.max;
const min = std.math.min;

const u16BorderVal = ~@as(u16,0);
const u16UnMappedLabel  =  u16BorderVal - 1;


pub fn fillImg2D(img:[]u8, nx:u32, ny:u32) !struct{img:[]u16, maxLabel:u16} {

  // Special label values
  // u16UnMappedLabel  = initial value for unvisited pixels
  // u16BorderVal = value for non-labeled regions
  
  // Initial conditions...
  // We need to know which sites are borders (and can't be merged) and which sites haven't been visited yet.
  // We can make both of these values `poisonVal` initially? But leave the border values as poison after visiting them.
  var res = try allocator.alloc(u16,img.len); // return this value, don't free
  for (res) |*v| v.* = u16BorderVal; // use border val for unvisited labels as well!
  for (img) |*v,i| {
    if (v.*>0) {
      res[i] = u16BorderVal; // non-label boundary value (always remapped to zero)
    }
  }

  // Algorithm internal state
  var currentLabel:?u16 = null;
  var currentMax:u16 = 0;

  var dj = try DisjointSets.init(64_000);
  // var map = std.AutoHashMap(u16,u16).init(allocator);
  // defer map.deinit();

  // First loop over the image. Label each pixel greedily, but build up a hash
  // map of pixels to remap.
  // print("Stage One:\n", .{});
  {
  var i:u32=0; while (i<ny):(i+=1){
  var j:u32=0; while (j<nx):(j+=1){

    const v = img[i*nx + j];

    // if v==0 we need to draw a label
    if (v==0){

      // Get set of all neib label values and their associated SetRoot.
      // Let's do boundary conditions by assuming a Border outside the image.
      const l1  = if (j>0)    res[i*nx + j-1]   else u16BorderVal;
      // const l2  = if (j<nx-1) res[i*nx + j+1]   else u16BorderVal;
      const l3  = if (i>0)    res[(i-1)*nx + j] else u16BorderVal;
      // const l4  = if (i<ny-1) res[(i+1)*nx + j] else u16BorderVal;

      // root is SetRoot of all neighbour labels
      // const root = dj.merge(l1,dj.merge(l2,dj.merge(l3,l4)));
      const root = dj.merge(l1,l3);
      
      if (currentLabel) |cl| {        // both currentLabel and root are valid. merge them and use the new root.
        currentLabel = dj.merge(cl,root);
      } else if (root < u16UnMappedLabel) { // currentLabel is void, but root is good. use root.
        currentLabel = root;
      } else {                        // currentLabel is void and root is poison. make a new highest label and use it.
        currentMax += 1;
        currentLabel = currentMax;
      }

      res[i*nx + j] = currentLabel.?;

    // Otherwise we don't need to draw a new label. Thus, we also
    // don't need to add any remapping to the hashmap. But we do need to 
    // unset the currentLabel (make sure it's null).
    } else {
      res[i*nx + j] = u16BorderVal;
      currentLabel = null;
    }
  }}}

  // try countPixelValues(res);
  // print("Stage Two:\n", .{});

  // try map.put(u16BorderVal, 0); // remap at the end. we want the stand-in value to be high to simplify the alg.
  // dj.map[u16BorderVal] = 0;

  // const maxval = im.minmax(u16,res)[1];
  // assert(maxval < u16UnMappedLabel);

  // try map.put(u16UnMappedLabel,  0); // remap at the end. we want the stand-in value to be high to simplify the alg.
  // Now loop over every pixel and lookup the label value. Remap if needed.
  {
  var i:u32=0; while (i<ny):(i+=1){
  var j:u32=0; while (j<nx):(j+=1){
    const label = res[i*nx + j];
    if (label==u16BorderVal){
      res[i*nx + j] = 0;  
    } else {
      res[i*nx + j] = dj.find(label);
    }
  }}}

  const ans = .{.img=res, .maxLabel=currentMax}; // TODO: why is this function able to infer the return type? i thought it couldn't do this?
  return ans;
}



test "imageToys. color square grid with fillImg2D()" {
  // fn fillImg2DSimple() !void {
  // print("\n", .{});

  var nx: u32 = 101;
  var ny: u32 = 101;

  // Make tic-tac-toe board image
  var img = try allocator.alloc(u32, nx * ny);
  defer allocator.free(img);
  for (img) |*v, i| {
    var ny_ = i/nx;
    var nx_ = i%nx;
    if (nx_%10==0 or ny_%10==0) v.* = 1 else v.* = 0;
  }

  // label the empty 0-valued spaces between bezier crossings
  var imgu8 = try allocator.alloc(u8, nx*ny);
  for (imgu8) |*v,i| v.* = @intCast(u8, img[i]);
  const _res = try fillImg2D(imgu8, @intCast(u16,nx) , @intCast(u16,ny) );
  const res = _res.img;
  const maxLabel = _res.maxLabel;
  print("MaxLabel = {}\n", .{maxLabel});

  try printPixelValueCounts(res);

  // std.sort.sort(u16 , res , {} , comptime std.sort.asc(u16)); // block sort
  // print("\n{d}\n",.{res[res.len-100..]});

  // Save the result as a greyscale image.
  for (imgu8) |*v,i| v.* = @intCast(u8, res[i]%256);
  try im.saveU8AsTGAGrey(imgu8, @intCast(u16, ny), @intCast(u16, nx), "fill_simple.tga");

  for (imgu8) |*v,i| v.* = @intCast(u8, img[i]*255);
  try im.saveU8AsTGAGrey(imgu8, @intCast(u16, ny), @intCast(u16, nx), "fill_simple_board.tga");


  // // Print the image in a grid
  // for (res) |*v,i| {
  //   if (i%101==0) print("\n",.{});
  //   print("{d} ",.{v.*});
  // }
}

// Disjoint Set data structure. see [notes.md @ Fri Sep  3]
// Used to keep track of labels in `fillImg2D()` 
const DisjointSets = struct {

    const This = @This();
    const Elem = u16;
    const SetRoot = u16;
    const poisonVal  = u16BorderVal; 
    const unknownVal = u16BorderVal - 1;
    const rootParent = u16BorderVal - 2;
    // labels in use are [1..rootParent-1]

    map : []Elem,
    nElems : usize,

    pub fn init(n:usize) !This {
        // const _map = try allocator.alloc(Node, ids.len);
        const _map = try allocator.alloc(Elem, n);
        // All elements are in their own set to start off.
        for (_map) |*v| v.* = rootParent;
        return This{.map=_map, .nElems=n};
    }

    // Double the number of labels in the total set. [Uses 👇 allocator.resize()]
    pub fn grow(this:This) !void {
        this.nElems *= 2;
        this.map = try allocator.resize(this.map, this.nElems);
        for (this.map[this.nElems/2 ..]) |*v| v.* = rootParent;
    }    
    // Return the SetRoot label associated with an arbitrary label.
    pub fn find(this:This, elem:Elem) SetRoot {
        var current = elem;
        var parent  = this.map[@intCast(usize,elem)];
        while (parent != rootParent) {
            current = parent;
            parent  = this.map[@intCast(usize,current)];
        }
        return current;
    }

    // Find sets associated with elements s1 and s2 and merge them together, returning the lower-valued SetRoot label.
    // Reserve poisonVal for an element that must always be in it's own singleton set and cannot be merged.
    pub fn merge(this:This, s1:Elem, s2:Elem) SetRoot {

        // return early if at least one value is invalid
        if (s1>=rootParent and s2<rootParent) return s2;
        if (s2>=rootParent and s1<rootParent) return s1;
        if (s1>=rootParent and s2>=rootParent) return s1;

        // else they must be valid
        const root1 = this.find(s1);
        const root2 = this.find(s2);
        if (root1 == root2) return root1;
        if (root1 < root2) {
            this.map[@intCast(usize,root2)] = root1;
            return root1;
        } else {
            this.map[@intCast(usize,root1)] = root2;
            return root2;  
        }
    }

    // TODO: Implement a function to create Sets of Values (instead of returning the SetRoot). Should be more efficient than iterating find() over all keys.
};


test "imageToys. DisjointSets datastructure basics" {
  // fn testDisjointSets() !void {
  var dj = try DisjointSets.init(1000);
  print("{}\n", .{dj.find(5)});
  _ = dj.merge(5,9);
  _ = dj.merge(3,8);
  _ = dj.merge(3,5);
  print("{}\n", .{dj.find(3)});
  print("{}\n", .{dj.find(5)});
  print("{}\n", .{dj.find(8)});
  print("{}\n", .{dj.find(9)});
}





// remap each element in the greyscale array to a random RGB value (returned array is 4x size)
pub fn randomColorLabels(comptime T:type, lab:[]T) ![]u8 {

  const mima = im.minmax(T,lab);
  var rgbmap = try allocator.alloc(u8, 3 * (@intCast(u16,mima[1]) + 1) );
  defer allocator.free(rgbmap);
  for (rgbmap) |*v| v.* = rando.int(u8);

  // map 0 to black
  rgbmap[0] = 0;
  rgbmap[1] = 0;
  rgbmap[2] = 0;
  rgbmap[3] = 255;

  // make new recolored image
  var rgbImg = try allocator.alloc(u8, 4*lab.len);
  for (lab) |*v,i| {
    const l  = 3 * @intCast(u16,v.*);
    rgbImg[4*i + 0] = rgbmap[l + 0];
    rgbImg[4*i + 1] = rgbmap[l + 1];
    rgbImg[4*i + 2] = rgbmap[l + 2];
    rgbImg[4*i + 3] = 255;
  }

  return rgbImg;
}

// iterate through all hashmap values and print "key -> val"
fn printHashMap(map:std.AutoHashMap(u16,u16)) void {
  {var i:u16=0; while(i<20):(i+=1){
    const x = map.get(i);
    print("{} → {}\n", .{i,x});
    }}
}

pub fn printPixelValueCounts(img: []u16) !void{
  var map = std.AutoHashMap(u16,u32).init(allocator); // label -> count
  defer map.deinit();
  var maxLabel:u16 = 0;
  for (img) |*v| {
    if (v.* > maxLabel) maxLabel=v.*;
    const count = map.get(v.*);
    if (count) |ct| {try map.put(v.*,ct+1);}
    else {try map.put(v.*,1);}
  }

  // Print out the label counts
  print("histogram\n", .{});
  {var i:u16=0; while(i<maxLabel):(i+=1){
    const count = map.get(i);
    if (count) |ct| {
      if (ct>0) print("{d} -> {d}\n", .{i,ct});
    }
  }}
}


// 2D disk sampling. caller owns returned memory.
// pub fn random2DPoints(npts:u32) ![]Vec2 {
// pub fn main() !void {
test "imageToys. 2d stratified sampling and radial distribution function" {

  const npts:u32 = 120*100;
  const img = try Img2D([4]u8).init(1200,1000);

  // clear screen
  for (img.img) |*v| v.* = .{0,0,0,255};

  var pts = try allocator.alloc(Vec2, npts);


  {var i:u32=0; 
    while (i<npts) : (i+=1) {
      const x = rando.uintLessThan(u31,@intCast(u31,img.nx));
      const y = rando.uintLessThan(u31,@intCast(u31,img.ny));
      
      pts[@intCast(u32,i)] = .{@intToFloat(f32,x) , @intToFloat(f32,y)};

      // const idx = x + y*img.nx;
      // img.img[idx] = .{128,0,0,255};
      draw.drawCircle([4]u8, img, x, y, 3, .{128,128,0,255});
    }}
  try im.saveRGBA(img,"randomptsEvenDistribution.tga");

  // clear screen
  for (img.img) |*v| v.* = .{0,0,0,255};

  const nx = 120;
  // const nx_f32 = @as(f32,nx);
  // const ny = 100;

  {var i:i32=0; 
    while (i<npts) : (i+=1) {
      const x = @mod(i,nx) * 10  + 5    ; 
      const y = @divFloor(i,nx) * 12  + 6 ;
      // print("x,y = {d} , {d}\n", .{x,y});
      const x2 = x + rando.intRangeAtMost(i32,-2,2);
      const y2 = y + rando.intRangeAtMost(i32,-3,3);

      // pts[@intCast(u32,i)] = .{@intToFloat(f32,x2) , @intToFloat(f32,y2)};

      // const idx = x + y*img.nx;
      // img.img[idx] = .{128,0,0,255};

      // const Myrgba = packed struct {r:u8,g:u8,b:u8,a:u8};
      var rgba = @bitCast([4]u8,@bitReverse(i32,i));
      // rgba[3] = 255;
      rgba = .{255,255,255,255};

      // drawCircle([4]u8, img, x2, y2, 1, .{128,128,0,255});
      draw.drawCircle([4]u8, img, x2, y2, 1, rgba);
    }}
  try im.saveRGBA(img,"randomptsGridWRandomDisplacement.tga");


  try radialDistribution(pts);
  // try drawPoints2D(Vec2, points2D:[]T, picname:Str, lines:?bool)
  // const floatpic = try Img2D(f32).init(1000,1000);
  // drawVerts(floatpic , verts)
}


const absInt = std.math.absInt;
// const assert = std.debug.assert;


pub fn radialDistribution(pts:[]Vec2) !void {

  _ = pts.len; // num particles
  // const pts2f32 = std.mem.bytesAsSlice([2]f32, std.mem.sliceAsBytes(pts));
  // `pts` natural coordinates...
  const bounds = geo.bounds2(pts);
  const midpoint = Vec2{(bounds[0][0]+bounds[1][0])/2 , (bounds[0][1]+bounds[1][1])/2};
  const width  = bounds[1][0] - bounds[0][0];
  const height = bounds[1][1] - bounds[0][1];

  const pt2pic = struct {
    pub fn f(pt:Vec2 , prm:anytype) [2]u32 {
      const pt_pix = (pt-prm.m) / Vec2{prm.h , prm.h} * Vec2{500,500} + Vec2{1000*prm.r, 1000};
      const res    = .{@floatToInt(u32,pt_pix[0]) , @floatToInt(u32,pt_pix[1])};
      return res;
    }
  }.f;

  // `count` will bin the points onto the pixel grid. Then `pic` will have a color.
  const count = try Img2D(u32).init(@floatToInt(u32, width/height*2000), 2000);
  for (count.img) |*c| c.* = 0;

  for (pts) |pt0,i| {
  for (pts) |pt1,j| {
    if (i==j) continue;
    const prm = .{.h=height , .m=midpoint , .r=width/height};
    const pix = pt2pic(pt1-pt0 , prm);
    count.img[pix[0] + 2000*pix[1]] += 1;
  }
  }

  const pic = try Img2D([4]u8).init(count.nx , count.ny);
  for (pic.img) |*rgba| rgba.* = .{0,0,0,255};
  // pic.img[pix[0] + 2000*pix[1]] = .{255,255,255,255};
  const count_max = std.mem.max(u32,count.img);
  for (pic.img) |*rgba,i| {
    // const val = @intToFloat(f32,count.img[i]) / @intToFloat(f32,count_max);
    const v = @intCast(u8, @divFloor(count.img[i]*255, count_max));
    rgba.* = .{v,v,v,255};
  }

  try im.saveRGBA(pic,"radialDisticution.tga");
}



// USES 👇 BRESHENHAM USES 👇 BRESHENHAM USES 👇 BRESHENHAM USES 👇 BRESHENHAM USES 👇 BRESHENHAM USES 👇 BRESHENHAM 
// USES 👇 BRESHENHAM USES 👇 BRESHENHAM USES 👇 BRESHENHAM USES 👇 BRESHENHAM USES 👇 BRESHENHAM USES 👇 BRESHENHAM 
// USES 👇 BRESHENHAM USES 👇 BRESHENHAM USES 👇 BRESHENHAM USES 👇 BRESHENHAM USES 👇 BRESHENHAM USES 👇 BRESHENHAM 

test "imageToys. render BoxPoly and filled cube" {

  // try mkdirIgnoreExists("filledCube");

  // const poly = mesh.BoxPoly.createAABB(.{0,0,0} , .{10,12,14});
  var cam = try PerspectiveCamera.init(.{100,100,100}, .{5,5,5}, 401, 301, null,);
  // var name = try allocator.alloc(u8,40);
  // const container = Img2D(f32){.img=cam.screen , .nx=cam.nxPixels , .ny=cam.nyPixels};
  var nx:u32=401; var ny:u32=301;

  cc.bres.init(&cam.screen[0],&ny,&nx);
  // print("\nvertices:\n{d:.2}",.{poly.vs});

  // const polyImage = try allocator.alloc(f32,10*12*14);
  var img = Img3D(f32){
    .img = try allocator.alloc(f32,10*12*14) ,
    .nx  = 14,
    .ny  = 12,
    .nz  = 10,
  };
  
  for (img.img) |*v,i| v.* = @intToFloat(f32,i);

  // print("minmax cam screen: {d}\n", .{im.minmax(f32,img.img)});

  perspectiveProjection2(img,&cam);
  try im.saveF32AsTGAGreyNormed(cam.screen, 301, 401, "filledCube.tga");

  // for (poly.es) |e,i| {
  //   // if (e[0]!=0) continue;
  //   const p0 = cam.world2pix(poly.vs[e[0]]);
  //   const p1 = cam.world2pix(poly.vs[e[1]]);
  //   // plotLine(f32,container,p0.x,p0.y,p1.x,p1.y,@intToFloat(f32,1));
  //   cc.bres.plotLine(@intCast(i32,p0.x) , @intCast(i32,p0.y) , @intCast(i32,p1.x) , @intCast(i32,p1.y));
  //   print("e={d}\n",.{e});
  //   print("p0={d:.2}\n",.{p0});
  //   print("p1={d:.2}\n",.{p1});
  //   name = try std.fmt.bufPrint(name, "filledCube/sides{:0>4}.tga", .{i});
  //   try im.saveF32AsTGAGreyNormed(allocator, cam.screen, 301, 401, name);
  // }
}


test "imageToys. spinning lorenz attractor" {
  // pub fn main() !void {
  var state0 = Vec3{1.0, 1.0, 1.0};
  const dt = 0.001;
  var times  = nparange(f32,0,40,dt);
  // var state1 = state0;

  var history = try allocator.alloc(Vec3,times.len);
  defer allocator.free(history);

  // simplest possible integration
  for (times) |_,i| {
    history[i] = state0;
    const dv = lorenzEquation(state0,10,28,8/3);
    state0 += dv * Vec3{dt,dt,dt};
  }

  // Draw Lorenz Attractor in 2D in three xy,xz,yz projections
  var pts:[2*times.len]f32 = undefined;
  for (history) |v,i| {
    pts[2*i] = v[0];
    pts[2*i+1] = v[1];
  }
  try draw.drawPoints2D(f32,pts[0..],"lorenzXY.tga",true);
  for (history) |v,i| {
    pts[2*i] = v[0];
    pts[2*i+1] = v[2];
  }
  try draw.drawPoints2D(f32,pts[0..],"lorenzXZ.tga",true);
  for (history) |v,i| {
    pts[2*i] = v[1];
    pts[2*i+1] = v[2];
  }
  try draw.drawPoints2D(f32,pts[0..],"lorenzYZ.tga",true);


  try mkdirIgnoreExists("lorenzSpin");

  // focus on center of manifold 3D bounding box, define camera trajectory in 3D
  const bds = geo.bounds3(history); 
  const focus = (bds[1] + bds[0]) / Vec3{2,2,2};
  const spin = sphereTrajectory();
  var name = try allocator.alloc(u8,40);
  for (spin) |campt,j| {

    // project from 3d->2d with perspective, draw lines in 2d, then save
    var cam = try PerspectiveCamera.init(campt*Vec3{300,300,300}, focus, 1600, 900, null, );
    defer cam.deinit();
    cc.bres.init(&cam.screen[0],&cam.nyPixels,&cam.nxPixels);
    for (history) |pt,i| {
      const px = cam.world2pix(pt);
      cc.bres.setPixel(@intCast(i32,px.x),@intCast(i32,px.y));
      if (i>0) {
        const px0 = cam.world2pix(history[i-1]);
        cc.bres.plotLine(@intCast(i32,px0.x),@intCast(i32,px0.y),@intCast(i32,px.x),@intCast(i32,px.y));
      }
    }
    name = try std.fmt.bufPrint(name, "lorenzSpin/img{:0>4}.tga", .{j});
    try im.saveF32AsTGAGreyNormedCam(cam, name);

  }
}


// Writes pixel values on curve to 1.0
pub fn randomBezierCurves(img:*Img2D(f32) , ncurves:u16) void {

  // Set bresenham global state.
  var nx = img.nx;
  var ny = img.ny;
  cc.bres.init(&img.img[0], &ny, &nx);
  
  // Generate 100 random Bezier curve segments and draw them in the image.
  {var i: u16 = 0;
  while (i < ncurves) : (i += 1) {
    const x0 = @intCast(i32, rando.uintLessThan(u32, nx));
    const y0 = @intCast(i32, rando.uintLessThan(u32, ny));
    const x1 = @intCast(i32, rando.uintLessThan(u32, nx));
    const y1 = @intCast(i32, rando.uintLessThan(u32, ny));
    const x2 = @intCast(i32, rando.uintLessThan(u32, nx));
    const y2 = @intCast(i32, rando.uintLessThan(u32, ny));
    cc.bres.plotQuadBezier(x0, y0, x1, y1, x2, y2);
  }}
}

test "imageToys. color random Bezier curves with fillImg2D()" {
  // fn testFillImg2D() !void {
  // print("\n", .{});

  var nx: u32 = 1900;
  var ny: u32 = 1024;

  // Initialize memory to 0. Set bresenham global state.
  var img = try allocator.alloc(f32, nx * ny);
  defer allocator.free(img);
  for (img) |*v| v.* = 0;
  cc.bres.init(&img[0], &ny, &nx);

  // Generate 100 random Bezier curve segments and draw them in the image.
  {var i: u16 = 0;
  while (i < 100) : (i += 1) {
    const x0 = @intCast(i32, rando.uintLessThan(u32, nx));
    const y0 = @intCast(i32, rando.uintLessThan(u32, ny));
    const x1 = @intCast(i32, rando.uintLessThan(u32, nx));
    const y1 = @intCast(i32, rando.uintLessThan(u32, ny));
    const x2 = @intCast(i32, rando.uintLessThan(u32, nx));
    const y2 = @intCast(i32, rando.uintLessThan(u32, ny));
    cc.bres.plotQuadBezier(x0, y0, x1, y1, x2, y2);
  }}

  // label the empty 0-valued spaces between bezier crossings
  var imgu8 = try allocator.alloc(u8, nx*ny);
  for (imgu8) |*v,i| v.* = @floatToInt(u8, img[i]);
  const _res = try fillImg2D(imgu8, @intCast(u16,nx) , @intCast(u16,ny) );
  const res = _res.img;
  const maxLabel = _res.maxLabel;
  print("MaxLabel = {}\n", .{maxLabel});

  // try countPixelValues(res);
  try printPixelValueCounts(res);

  // std.sort.sort(u16 , res , {} , comptime std.sort.asc(u16)); // block sort
  // print("\n{d}\n",.{res[res.len-100..]});

  // Save the result as a greyscale image.
  for (imgu8) |*v,i| v.* = @intCast(u8, res[i]%256);
  // try im.saveU8AsTGAGrey(allocator, imgu8, @intCast(u16, ny), @intCast(u16, nx), "fill_curve.tga");

  const rgbImg = try randomColorLabels(u8, imgu8);
  defer allocator.free(rgbImg);

  try im.saveU8AsTGA(rgbImg, @intCast(u16,ny), @intCast(u16,nx), "rgbFillImg.tga");
}

test "imageToys. save img of Random Bezier Curves" {
  print("\n", .{});

  // Initialize memory to 0. 
  var img = Img2D(f32){
    .img = try allocator.alloc(f32, 1800 * 1200),
    .nx = 1800,
    .ny = 1200,
  };
  defer allocator.free(img.img);
  for (img.img) |*v| v.* = 0;
  

  // 10 Random Bezier Curves
  randomBezierCurves(&img , 10);

  // Convert to u8 and save as grescale image.
  // QUESTION: how can I make this memory `const` ? (or at least, const after the initialization?)
  var data = try allocator.alloc(u8, img.nx * img.ny);
  defer allocator.free(data);
  for (data) |*v, i| v.* = @floatToInt(u8, img.img[i] * 255) ;

  try im.saveU8AsTGAGrey(data, @intCast(u16, img.ny), @intCast(u16, img.nx), "randomBezierCurves.tga");
}