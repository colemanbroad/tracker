// A datastructure that efficiently encodes a sparsely connected graph

// convert triangles into vertex-vertex mapping
// each vertex has it's own array of size [10]u32 . array[0] stores #elements array[1..10] store neib id's

pub fn V2VGraph(comptime depth: u8) type {
    return struct {
        const Self = @This();
        map: [][depth]?u32,

        pub fn init(al: Allocator, na: u32) Self {
            var map = try al.alloc([depth]?u32, na);
            // defer allocator.free(map);
            for (map) |*v| v.* = .{null} ** depth; // id, id, id, id, ...
            return Self{ .map = map };
        }

        // Be careful. We're iterating through triangles, so we see each interior edge TWICE!
        // This loop will de-duplicate edges.
        // for each vertex `v` from triangle `tri` with neighbour vertex `v_neib` we loop over
        // all existing neibs to see if `v_neib` already exists. if it doesn't we add it.
        // WARNING: this will break early once we hit `null`. This is fine as long as the array is front-packed like
        // [value value value null null ...]

        pub fn fromTriangles(self: Self, triangles: [][3]Vec2) void {
            for (triangles) |tri| {
                for (tri) |v, i| {
                    outer: for ([3]u32{ 0, 1, 2 }) |j| {
                        if (i == j) continue;
                        const v_neib = tri[j];
                        for (self.map[v]) |v_neib_existing, k| {
                            if (v_neib_existing == null) {
                                self.map[v][k] = v_neib;
                                self.dist[v][k] = dist(Pts, va[v], va[v_neib]); // squared euclidean
                                continue :outer;
                            }
                            if (v_neib_existing.? == v_neib) continue :outer;
                        }
                    }
                }
            }
        }

        // average distance between neighbouring points
        pub fn avgDist(self: Self) f32 {
            var ad: f32 = 0;
            var count: u32 = 0;
            for (self.dist) |nnd| {
                for (nnd) |dq| {
                    if (dq) |d| {
                        ad += d;
                        count += 1;
                    }
                }
            }
            return ad / @intToFloat(f32, count);
        }
    };
}
