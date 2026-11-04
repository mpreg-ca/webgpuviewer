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

fn catmull_rom_weights(t: f32) -> array<f32, 4> {
    let t2 = t * t;
    let t3 = t2 * t;

    return array<f32, 4>(
        -0.5 * t3 + t2 - 0.5 * t,
         1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
         0.5 * t3 - 0.5 * t2
    );
}

fn catmull_rom_load(pos: vec2<i32>, b: vec4<i32>) -> vec4<f32> {
    let c = to_linear_exact(load(clamp(pos, b.xy, b.zw)));
    return vec4<f32>(c.rgb * c.a, c.a);
}

fn catmull_rom_magnify(src_pos: vec2<f32>, b: vec4<i32>) -> vec4<f32> {
    let pixel_coord = src_pos - 0.5;
    let start = vec2<i32>(floor(pixel_coord)) - 1;
    let f = fract(pixel_coord);
    let wx = catmull_rom_weights(f.x);
    let wy = catmull_rom_weights(f.y);

    let r0 = catmull_rom_load(start + vec2<i32>(0, 0), b) * wx[0]
           + catmull_rom_load(start + vec2<i32>(1, 0), b) * wx[1]
           + catmull_rom_load(start + vec2<i32>(2, 0), b) * wx[2]
           + catmull_rom_load(start + vec2<i32>(3, 0), b) * wx[3];
    let r1 = catmull_rom_load(start + vec2<i32>(0, 1), b) * wx[0]
           + catmull_rom_load(start + vec2<i32>(1, 1), b) * wx[1]
           + catmull_rom_load(start + vec2<i32>(2, 1), b) * wx[2]
           + catmull_rom_load(start + vec2<i32>(3, 1), b) * wx[3];
    let r2 = catmull_rom_load(start + vec2<i32>(0, 2), b) * wx[0]
           + catmull_rom_load(start + vec2<i32>(1, 2), b) * wx[1]
           + catmull_rom_load(start + vec2<i32>(2, 2), b) * wx[2]
           + catmull_rom_load(start + vec2<i32>(3, 2), b) * wx[3];
    let r3 = catmull_rom_load(start + vec2<i32>(0, 3), b) * wx[0]
           + catmull_rom_load(start + vec2<i32>(1, 3), b) * wx[1]
           + catmull_rom_load(start + vec2<i32>(2, 3), b) * wx[2]
           + catmull_rom_load(start + vec2<i32>(3, 3), b) * wx[3];
    return r0 * wy[0] + r1 * wy[1] + r2 * wy[2] + r3 * wy[3];
}

fn catmull_rom_kernel(x: f32) -> f32 {
    let a = abs(x);
    if (a < 1.0) {
        return 1.5 * a * a * a - 2.5 * a * a + 1.0;
    }
    if (a < 2.0) {
        return -0.5 * a * a * a + 2.5 * a * a - 4.0 * a + 2.0;
    }
    return 0.0;
}

// Stretched over a footprint wider than a texel, so shrinking low-passes instead of aliasing.
fn catmull_rom_scaled(center: vec2<f32>, scale: vec2<f32>, b: vec4<i32>) -> vec4<f32> {
    let s = max(scale, vec2<f32>(1.0));
    // Texel i, centred at i + 0.5, is in reach within two scaled texels.
    let first = vec2<i32>(ceil(center - 0.5 - 2.0 * s));
    let last = vec2<i32>(floor(center - 0.5 + 2.0 * s));

    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var y: i32 = first.y; y <= last.y; y++) {
        let wy = catmull_rom_kernel((f32(y) + 0.5 - center.y) / s.y);
        for (var x: i32 = first.x; x <= last.x; x++) {
            let w = wy * catmull_rom_kernel((f32(x) + 0.5 - center.x) / s.x);
            sum += catmull_rom_load(vec2<i32>(x, y), b) * w;
            total += w;
        }
    }
    return sum / max(total, 0.0001);
}

fn resolve(src_start: vec2<f32>, scale: vec2<f32>, b: vec4<i32>) -> vec4<f32> {
    let center = src_start + 0.5 * scale;
    var c: vec4<f32>;
    if (max(scale.x, scale.y) <= 1.0) {
        c = catmull_rom_magnify(center, b);
    } else {
        c = catmull_rom_scaled(center, scale, b);
    }
    return to_srgb_exact(vec4<f32>(select(vec3<f32>(0.0), c.rgb / c.a, c.a > 0.0), c.a));
}
