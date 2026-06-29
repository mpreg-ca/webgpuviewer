package ca.mpreg.webgpuviewer.transitions

val ImageShaderSingle = """
struct Uniforms {
    offset: vec2<f32>,
    scale: f32,
    dst_width: f32,
    dst_height: f32,
    padding0: f32,
    padding1: f32,
    padding2: f32,
}

@group(0) @binding(0) var<uniform> transform: Uniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), // Bottom-left
        vec2<f32>( 3.0, -1.0), // Far-right
        vec2<f32>(-1.0,  3.0)  // Far-top
    );

    var out: VertexOutput;
    let p = pos[vertex_index];
    out.position = vec4<f32>(p, 0.0, 1.0);

    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
    return out;
}

fn to_linear_exact(srgb: vec4<f32>) -> vec4<f32> {
    let c = max(srgb.rgb, vec3<f32>(0.0));
    let lower = c / vec3<f32>(12.92);
    let higher = pow((c + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let cond = c <= vec3<f32>(0.04045);
    return vec4(select(higher, lower, cond), srgb.a);
}

fn to_srgb_exact(linear_rgb: vec4<f32>) -> vec4<f32> {
    let c = max(linear_rgb.rgb, vec3<f32>(0.0));
    let lower = c * vec3<f32>(12.92);
    let higher = vec3<f32>(1.055) * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    let cond = c <= vec3<f32>(0.0031308);
    return vec4(select(higher, lower, cond), linear_rgb.a);
}

fn catmull_rom_weights(t: f32) -> array<f32, 4> {
    let t2 = t * t;
    let t3 = t2 * t;

    return array<f32, 4>(
        -0.5 * t3 + t2 - 0.5 * t,          // Weight 0 (Negative lobe)
         1.5 * t3 - 2.5 * t2 + 1.0,        // Weight 1 (Primary influence)
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,    // Weight 2 (Primary influence)
         0.5 * t3 - 0.5 * t2               // Weight 3 (Negative lobe)
    );
}

fn catmull_rom_fast_unrolled(
    tex: texture_2d<f32>,
    p_start: vec2<i32>,
    wx: array<f32, 4>,
    wy: array<f32, 4>
) -> vec4<f32> {
    let r0 = to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x,     p_start.y), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 1, p_start.y), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 2, p_start.y), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 3, p_start.y), 0)) * wx[3];
    let r1 = to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x,     p_start.y + 1), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 1, p_start.y + 1), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 2, p_start.y + 1), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 3, p_start.y + 1), 0)) * wx[3];
    let r2 = to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x,     p_start.y + 2), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 1, p_start.y + 2), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 2, p_start.y + 2), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 3, p_start.y + 2), 0)) * wx[3];
    let r3 = to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x,     p_start.y + 3), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 1, p_start.y + 3), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 2, p_start.y + 3), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 3, p_start.y + 3), 0)) * wx[3];

    return r0 * wy[0] + r1 * wy[1] + r2 * wy[2] + r3 * wy[3];
}

fn load_safe_linear(pos: vec2<i32>, max_coord: vec2<i32>) -> vec4<f32> {
    if (pos.x >= 0 && pos.x <= max_coord.x && pos.y >= 0 && pos.y <= max_coord.y) {
        return to_linear_exact(textureLoad(src_tex, pos, 0));
    }
    return vec4<f32>(0.0);
}

fn catmull_rom_slow_unrolled(
    start_i: vec2<i32>,
    max_coord: vec2<i32>,
    wx: array<f32, 4>,
    wy: array<f32, 4>
) -> vec4<f32> {
    let r0 = load_safe_linear(vec2<i32>(start_i.x,     start_i.y), max_coord) * wx[0]
           + load_safe_linear(vec2<i32>(start_i.x + 1, start_i.y), max_coord) * wx[1]
           + load_safe_linear(vec2<i32>(start_i.x + 2, start_i.y), max_coord) * wx[2]
           + load_safe_linear(vec2<i32>(start_i.x + 3, start_i.y), max_coord) * wx[3];
    let r1 = load_safe_linear(vec2<i32>(start_i.x,     start_i.y + 1), max_coord) * wx[0]
           + load_safe_linear(vec2<i32>(start_i.x + 1, start_i.y + 1), max_coord) * wx[1]
           + load_safe_linear(vec2<i32>(start_i.x + 2, start_i.y + 1), max_coord) * wx[2]
           + load_safe_linear(vec2<i32>(start_i.x + 3, start_i.y + 1), max_coord) * wx[3];
    let r2 = load_safe_linear(vec2<i32>(start_i.x,     start_i.y + 2), max_coord) * wx[0]
           + load_safe_linear(vec2<i32>(start_i.x + 1, start_i.y + 2), max_coord) * wx[1]
           + load_safe_linear(vec2<i32>(start_i.x + 2, start_i.y + 2), max_coord) * wx[2]
           + load_safe_linear(vec2<i32>(start_i.x + 3, start_i.y + 2), max_coord) * wx[3];
    let r3 = load_safe_linear(vec2<i32>(start_i.x,     start_i.y + 3), max_coord) * wx[0]
           + load_safe_linear(vec2<i32>(start_i.x + 1, start_i.y + 3), max_coord) * wx[1]
           + load_safe_linear(vec2<i32>(start_i.x + 2, start_i.y + 3), max_coord) * wx[2]
           + load_safe_linear(vec2<i32>(start_i.x + 3, start_i.y + 3), max_coord) * wx[3];
    return r0 * wy[0] + r1 * wy[1] + r2 * wy[2] + r3 * wy[3];
}

fn textureSampleCatmullRom(uv: vec2<f32>) -> vec4<f32> {
    let tex_size_u = textureDimensions(src_tex);
    let tex_size = vec2<f32>(tex_size_u);
    let pixel_coord = uv * tex_size - 0.5;
    let base_coord = vec2<i32>(floor(pixel_coord));
    let f = fract(pixel_coord);

    let wx = catmull_rom_weights(f.x);
    let wy = catmull_rom_weights(f.y);
    let max_coord = vec2<i32>(tex_size_u) - 1;

    let start_i = base_coord - vec2<i32>(1); // Top-left
    let end_i   = base_coord + vec2<i32>(2); // Bottom-right

    let canvas_in_bounds = start_i.x >= 0 && start_i.y >= 0 && end_i.x <= max_coord.x && end_i.y <= max_coord.y;

    var final_color_linear = vec4<f32>(0.0);

    let p_start = start_i;

    final_color_linear = catmull_rom_fast_unrolled(src_tex, p_start, wx, wy);

    return clamp(to_srgb_exact(final_color_linear), vec4(0.0), vec4(1.0));
}

fn downsample(src_start: vec2<f32>, scale: vec2<f32>) -> vec4<f32> {
    let src_size = textureDimensions(src_tex);

    let src_size_f = vec2<f32>(src_size);

    let src_end = src_start + vec2<f32>(1) * scale;

    let start_i = vec2<i32>(clamp(floor(src_start), vec2<f32>(0.0), src_size_f));
    let end_i   = vec2<i32>(clamp(ceil(src_end), vec2<f32>(0.0), src_size_f));
    
    var color_sum = vec4<f32>(0.0);
    var weight_sum = 0.0;

    for (var y: i32 = start_i.y; y < end_i.y; y++) {
        let y_f = f32(y);
        let y_overlap = max(0.0, min(y_f + 1.0, src_end.y) - max(y_f, src_start.y));

        for (var x: i32 = start_i.x; x < end_i.x; x++) {
            let x_f = f32(x);
            let x_overlap = max(0.0, min(x_f + 1.0, src_end.x) - max(x_f, src_start.x));
            let weight = x_overlap * y_overlap;
            var texel = textureLoad(src_tex, vec2<i32>(x, y), 0);
            texel = to_linear_exact(texel);
            color_sum += texel * weight;
            weight_sum += weight;
        }
    }
    var col = color_sum / weight_sum;
    return to_srgb_exact(col);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dst_size_f = vec2<f32>(transform.dst_width, transform.dst_height);
    let src_size_f = vec2<f32>(textureDimensions(src_tex));

    let frag_coord = in.position.xy;

    let scale_factor = 1.0 / transform.scale;
    let scale_vec = vec2<f32>(scale_factor);
    let offset = transform.offset;

    if (scale_factor > 1.0) {
        let src_start = frag_coord * scale_vec - (offset * dst_size_f);
        return downsample(src_start, scale_vec);
    } else {
        let uv = (frag_coord * scale_vec - offset * dst_size_f) / src_size_f;
        return textureSampleCatmullRom(uv);
    }
}"""
