// Odd-symmetric as scRGB defines, so extended-sRGB negatives survive instead of being clamped.
fn to_linear_exact(srgb: vec4<f32>) -> vec4<f32> {
    let c = abs(srgb.rgb);
    let lower = c / vec3<f32>(12.92);
    let higher = pow((c + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let cond = c <= vec3<f32>(0.04045);
    return vec4(sign(srgb.rgb) * select(higher, lower, cond), srgb.a);
}

fn to_srgb_exact(linear_rgb: vec4<f32>) -> vec4<f32> {
    let c = abs(linear_rgb.rgb);
    let lower = c * vec3<f32>(12.92);
    let higher = vec3<f32>(1.055) * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    let cond = c <= vec3<f32>(0.0031308);
    return vec4(sign(linear_rgb.rgb) * select(higher, lower, cond), linear_rgb.a);
}

// Texel i's share of the footprint [lo, hi) on one axis. Past the bounds, the edge texel stands in.
fn box_overlap(i: i32, first: i32, last: i32, lo: f32, hi: f32) -> f32 {
    let f = f32(i);
    if (i == first) {
        return max(min(f + 1.0, hi) - lo, 0.0);
    }
    if (i == last) {
        return max(hi - max(f, lo), 0.0);
    }
    return 1.0;
}

// A footprint under a texel gives that texel, blended only where it crosses an edge.
fn resolve(src_start: vec2<f32>, scale: vec2<f32>, b: vec4<i32>) -> vec4<f32> {
    let src_end = src_start + scale;
    let start_i = clamp(vec2<i32>(floor(src_start)), b.xy, b.zw + 1);
    let end_i = clamp(vec2<i32>(ceil(src_end)), b.xy, b.zw + 1);

    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var y: i32 = start_i.y; y < end_i.y; y++) {
        let wy = box_overlap(y, start_i.y, end_i.y - 1, src_start.y, src_end.y);
        for (var x: i32 = start_i.x; x < end_i.x; x++) {
            let w = wy * box_overlap(x, start_i.x, end_i.x - 1, src_start.x, src_end.x);
            let c = to_linear_exact(load(vec2<i32>(x, y)));
            sum += vec4<f32>(c.rgb * c.a, c.a) * w;
            total += w;
        }
    }
    // Wholly past the bounds.
    if (total <= 0.0) {
        return load(clamp(vec2<i32>(floor(src_start)), b.xy, b.zw));
    }
    let avg = sum / total;
    return to_srgb_exact(vec4<f32>(select(vec3<f32>(0.0), avg.rgb / avg.a, avg.a > 0.0), avg.a));
}
