// This is a STUB. I want a datastructure that let's me keep a sparse mapping between two points sets
// instead of a dense NxM matrix .

// convert triangles into vertex-vertex mapping
// each vertex has it's own array of size [10]u32 . array[0] stores #elements array[1..10] store neib id's

pub fn ArraySet(comptime size:u8) type {

  const c0 = edges[t[0]*N];

  if (c0==0){
  // if we haven't added any neibs yet then just add them to the first two spots
    edges[t[0]*N + 1] = t[1];
    edges[t[0]*N + 2] = t[2];
  } else {
  // otherwise we need to check if they already exist.

    for (edges[t[0]*N..t[0]*N+c0]) |v0,j| {
      if (v0==t[1]) break;
    }

  }

}