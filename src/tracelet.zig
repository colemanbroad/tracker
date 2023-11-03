const std = @import("std");

/// Enables fast, nanosecond speed function profiling.
pub fn Tracer(comptime Ntrials: u32) type {
    return struct {
        const Self = @This();

        const TimePoint = struct {
            src_name: []const u8,
            delta_t: u64,
        };
        // timer: std.time.Timer = ;

        timer: std.time.Timer,
        v: [20 * Ntrials]TimePoint = undefined,
        idx: usize = 0,

        const Span = struct {
            tracer: *Self,
            src_name: []const u8,
            start: u64,

            pub fn stop(this: Span) void {
                this.tracer.append(.{
                    .src_name = this.src_name,
                    .delta_t = this.tracer.timer.read() - this.start,
                });
            }
        };

        pub fn init() !Self {
            return Self{
                .timer = try std.time.Timer.start(),
            };
        }

        pub fn start(this: *Self, src_name: []const u8) Span {
            return .{ .tracer = this, .src_name = src_name, .start = this.timer.read() };
        }

        pub fn append(this: *Self, tp: TimePoint) void {
            this.v[this.idx] = tp;
            this.idx += 1;
        }

        /// Group by source function name, then compute and print statistics on time intervals.
        pub fn analyze(this: Self, allocator: std.mem.Allocator) !void {
            var unique_functions = std.StringHashMap(void).init(allocator);
            for (this.v[0..this.idx]) |tp| {
                try unique_functions.put(tp.src_name, {});
            }
            var keyiter = unique_functions.keyIterator();
            while (keyiter.next()) |funcname| {
                var timespans: [Ntrials]f32 = undefined;
                var idx: u32 = 0;
                for (this.v[0..this.idx]) |tp| {
                    if (!std.mem.eql(u8, tp.src_name, funcname.*)) continue;
                    timespans[idx] = @as(f32, @floatFromInt(tp.delta_t));
                    idx += 1;
                }
                const stats = statistics(f32, timespans[0..idx]);
                std.debug.print("{s:<40} mean {d:>6.0} [ns]      stddev {d:>6.0} \n", .{ funcname.*, stats.mean, stats.stddev });
            }
        }
    };
}

const Stats = struct {
    mean: f64,
    min: f64,
    max: f64,
    mode: f64,
    median: f64,
    stddev: f64,

    pub fn format(self: Stats, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        try writer.print("{s:9} {d:>10.0}\n", .{ "mean", self.mean });
        try writer.print("{s:9} {d:>10.0}\n", .{ "median", self.median });
        try writer.print("{s:9} {d:>10.0}\n", .{ "mode", self.mode });
        try writer.print("{s:9} {d:>10.0}\n", .{ "min", self.min });
        try writer.print("{s:9} {d:>10.0}\n", .{ "max", self.max });
        try writer.print("{s:9} {d:>10.0}\n", .{ "std dev", self.stddev });

        try writer.writeAll("");
    }
};

fn statistics(comptime T: type, arr: []T) Stats {
    var s: Stats = undefined;

    s.mean = 0;
    s.stddev = 0;

    std.sort.heap(T, arr, {}, std.sort.asc(T));
    for (arr) |x| {
        s.mean += x;
        s.stddev += x * x;
    }
    s.mean /= @as(f32, @floatFromInt(arr.len));
    s.stddev /= @as(f32, @floatFromInt(arr.len));
    s.stddev -= s.mean * s.mean;
    s.stddev = @sqrt(s.stddev);

    s.min = arr[0];
    s.max = arr[arr.len - 1];
    s.median = arr[arr.len / 2];

    var max_count: u32 = 0;
    var current_count: u32 = 1;
    s.mode = arr[0];
    for (1..arr.len) |i| { // exclusive on right ?
        if (arr[i] != arr[i - 1]) {
            current_count = 1;
            continue;
        }

        current_count += 1;
        if (current_count <= max_count) continue;

        max_count = current_count;
        s.mode = arr[i];
    }

    return s;
}
