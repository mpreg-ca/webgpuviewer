package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPUTexture
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder

object TransitionFade : Transition() {
    override val code = """
struct Uniforms {
    offset1: vec2<f32>,
    scale1: f32,
    tile_size1: f32,
    tiles_width1: f32,
    tiles_height1: f32,
    offset2: vec2<f32>,
    scale2: f32,
    tile_size2: f32,
    tiles_width2: f32,
    tiles_height2: f32,
    dst_width: f32,
    dst_height: f32,
    blend: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var tex1_0: texture_2d<f32>;
@group(0) @binding(2) var tex1_1: texture_2d<f32>;
@group(0) @binding(3) var tex1_2: texture_2d<f32>;
@group(0) @binding(4) var tex1_3: texture_2d<f32>;
@group(0) @binding(5) var tex2_0: texture_2d<f32>;
@group(0) @binding(6) var tex2_1: texture_2d<f32>;
@group(0) @binding(7) var tex2_2: texture_2d<f32>;
@group(0) @binding(8) var tex2_3: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv1: vec2<f32>,
    @location(1) uv2: vec2<f32>,
};

fn tileLoad1(i: i32, pos: vec2<i32>) -> vec4<f32> {
    if (i == 0) { return textureLoad(tex1_0, pos, 0); }
    if (i == 1) { return textureLoad(tex1_1, pos, 0); }
    if (i == 2) { return textureLoad(tex1_2, pos, 0); }
    return textureLoad(tex1_3, pos, 0);
}

fn tileLoad2(i: i32, pos: vec2<i32>) -> vec4<f32> {
    if (i == 0) { return textureLoad(tex2_0, pos, 0); }
    if (i == 1) { return textureLoad(tex2_1, pos, 0); }
    if (i == 2) { return textureLoad(tex2_2, pos, 0); }
    return textureLoad(tex2_3, pos, 0);
}

fn totalDimensions1() -> vec2<u32> {
    let w = i32(u.tiles_width1);
    let h = i32(u.tiles_height1);
    if (w <= 0 || h <= 0) { return vec2<u32>(0u); }
    let dim0 = textureDimensions(tex1_0);
    var width = dim0.x;
    if (w > 1) { width += textureDimensions(tex1_1).x; }
    var height = dim0.y;
    if (h > 1) { height += textureDimensions(tex1_2).y; }
    return vec2<u32>(width, height);
}

fn totalDimensions2() -> vec2<u32> {
    let w = i32(u.tiles_width2);
    let h = i32(u.tiles_height2);
    if (w <= 0 || h <= 0) { return vec2<u32>(0u); }
    let dim0 = textureDimensions(tex2_0);
    var width = dim0.x;
    if (w > 1) { width += textureDimensions(tex2_1).x; }
    var height = dim0.y;
    if (h > 1) { height += textureDimensions(tex2_2).y; }
    return vec2<u32>(width, height);
}

fn totalLoad1(pos: vec2<i32>) -> vec4<f32> {
    let ts = i32(u.tile_size1);
    let tile_x = select(0, 1, pos.x >= ts);
    let tile_y = select(0, 1, pos.y >= ts);
    let idx = tile_y * 2 + tile_x;
    let pos0 = pos - vec2<i32>(tile_x, tile_y) * ts;
    return tileLoad1(idx, pos0);
}

fn totalLoad2(pos: vec2<i32>) -> vec4<f32> {
    let ts = i32(u.tile_size2);
    let tile_x = select(0, 1, pos.x >= ts);
    let tile_y = select(0, 1, pos.y >= ts);
    let idx = tile_y * 2 + tile_x;
    let pos0 = pos - vec2<i32>(tile_x, tile_y) * ts;
    return tileLoad2(idx, pos0);
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
        -0.5 * t3 + t2 - 0.5 * t,
         1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
         0.5 * t3 - 0.5 * t2
    );
}

// Image 1 sampling functions
fn catmull_rom_fast1(tex: texture_2d<f32>, p: vec2<i32>, wx: array<f32, 4>, wy: array<f32, 4>) -> vec4<f32> {
    let r0 = to_linear_exact(textureLoad(tex, vec2<i32>(p.x, p.y), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+1, p.y), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+2, p.y), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+3, p.y), 0)) * wx[3];
    let r1 = to_linear_exact(textureLoad(tex, vec2<i32>(p.x, p.y+1), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+1, p.y+1), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+2, p.y+1), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+3, p.y+1), 0)) * wx[3];
    let r2 = to_linear_exact(textureLoad(tex, vec2<i32>(p.x, p.y+2), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+1, p.y+2), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+2, p.y+2), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+3, p.y+2), 0)) * wx[3];
    let r3 = to_linear_exact(textureLoad(tex, vec2<i32>(p.x, p.y+3), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+1, p.y+3), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+2, p.y+3), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+3, p.y+3), 0)) * wx[3];
    return r0 * wy[0] + r1 * wy[1] + r2 * wy[2] + r3 * wy[3];
}

fn load_safe_linear1(pos: vec2<i32>, max_coord: vec2<i32>) -> vec4<f32> {
    if (pos.x >= 0 && pos.x <= max_coord.x && pos.y >= 0 && pos.y <= max_coord.y) {
        return to_linear_exact(totalLoad1(pos));
    }
    return vec4<f32>(0.0);
}

fn catmull_rom_slow1(s: vec2<i32>, m: vec2<i32>, wx: array<f32, 4>, wy: array<f32, 4>) -> vec4<f32> {
    let r0 = load_safe_linear1(vec2<i32>(s.x, s.y), m) * wx[0] + load_safe_linear1(vec2<i32>(s.x+1, s.y), m) * wx[1]
           + load_safe_linear1(vec2<i32>(s.x+2, s.y), m) * wx[2] + load_safe_linear1(vec2<i32>(s.x+3, s.y), m) * wx[3];
    let r1 = load_safe_linear1(vec2<i32>(s.x, s.y+1), m) * wx[0] + load_safe_linear1(vec2<i32>(s.x+1, s.y+1), m) * wx[1]
           + load_safe_linear1(vec2<i32>(s.x+2, s.y+1), m) * wx[2] + load_safe_linear1(vec2<i32>(s.x+3, s.y+1), m) * wx[3];
    let r2 = load_safe_linear1(vec2<i32>(s.x, s.y+2), m) * wx[0] + load_safe_linear1(vec2<i32>(s.x+1, s.y+2), m) * wx[1]
           + load_safe_linear1(vec2<i32>(s.x+2, s.y+2), m) * wx[2] + load_safe_linear1(vec2<i32>(s.x+3, s.y+2), m) * wx[3];
    let r3 = load_safe_linear1(vec2<i32>(s.x, s.y+3), m) * wx[0] + load_safe_linear1(vec2<i32>(s.x+1, s.y+3), m) * wx[1]
           + load_safe_linear1(vec2<i32>(s.x+2, s.y+3), m) * wx[2] + load_safe_linear1(vec2<i32>(s.x+3, s.y+3), m) * wx[3];
    return r0 * wy[0] + r1 * wy[1] + r2 * wy[2] + r3 * wy[3];
}

// Image 2 sampling functions
fn catmull_rom_fast2(tex: texture_2d<f32>, p: vec2<i32>, wx: array<f32, 4>, wy: array<f32, 4>) -> vec4<f32> {
    let r0 = to_linear_exact(textureLoad(tex, vec2<i32>(p.x, p.y), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+1, p.y), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+2, p.y), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+3, p.y), 0)) * wx[3];
    let r1 = to_linear_exact(textureLoad(tex, vec2<i32>(p.x, p.y+1), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+1, p.y+1), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+2, p.y+1), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+3, p.y+1), 0)) * wx[3];
    let r2 = to_linear_exact(textureLoad(tex, vec2<i32>(p.x, p.y+2), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+1, p.y+2), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+2, p.y+2), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+3, p.y+2), 0)) * wx[3];
    let r3 = to_linear_exact(textureLoad(tex, vec2<i32>(p.x, p.y+3), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+1, p.y+3), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+2, p.y+3), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p.x+3, p.y+3), 0)) * wx[3];
    return r0 * wy[0] + r1 * wy[1] + r2 * wy[2] + r3 * wy[3];
}

fn load_safe_linear2(pos: vec2<i32>, max_coord: vec2<i32>) -> vec4<f32> {
    if (pos.x >= 0 && pos.x <= max_coord.x && pos.y >= 0 && pos.y <= max_coord.y) {
        return to_linear_exact(totalLoad2(pos));
    }
    return vec4<f32>(0.0);
}

fn catmull_rom_slow2(s: vec2<i32>, m: vec2<i32>, wx: array<f32, 4>, wy: array<f32, 4>) -> vec4<f32> {
    let r0 = load_safe_linear2(vec2<i32>(s.x, s.y), m) * wx[0] + load_safe_linear2(vec2<i32>(s.x+1, s.y), m) * wx[1]
           + load_safe_linear2(vec2<i32>(s.x+2, s.y), m) * wx[2] + load_safe_linear2(vec2<i32>(s.x+3, s.y), m) * wx[3];
    let r1 = load_safe_linear2(vec2<i32>(s.x, s.y+1), m) * wx[0] + load_safe_linear2(vec2<i32>(s.x+1, s.y+1), m) * wx[1]
           + load_safe_linear2(vec2<i32>(s.x+2, s.y+1), m) * wx[2] + load_safe_linear2(vec2<i32>(s.x+3, s.y+1), m) * wx[3];
    let r2 = load_safe_linear2(vec2<i32>(s.x, s.y+2), m) * wx[0] + load_safe_linear2(vec2<i32>(s.x+1, s.y+2), m) * wx[1]
           + load_safe_linear2(vec2<i32>(s.x+2, s.y+2), m) * wx[2] + load_safe_linear2(vec2<i32>(s.x+3, s.y+2), m) * wx[3];
    let r3 = load_safe_linear2(vec2<i32>(s.x, s.y+3), m) * wx[0] + load_safe_linear2(vec2<i32>(s.x+1, s.y+3), m) * wx[1]
           + load_safe_linear2(vec2<i32>(s.x+2, s.y+3), m) * wx[2] + load_safe_linear2(vec2<i32>(s.x+3, s.y+3), m) * wx[3];
    return r0 * wy[0] + r1 * wy[1] + r2 * wy[2] + r3 * wy[3];
}

fn textureSampleCatmullRom1(uv: vec2<f32>) -> vec4<f32> {
    let tex_size_u = totalDimensions1();
    let tex_size = vec2<f32>(tex_size_u);
    let pixel_coord = uv * tex_size - 0.5;
    let base_coord = vec2<i32>(floor(pixel_coord));
    let f = fract(pixel_coord);
    let wx = catmull_rom_weights(f.x);
    let wy = catmull_rom_weights(f.y);
    let max_coord = vec2<i32>(tex_size_u) - 1;
    let ts = i32(u.tile_size1);
    let start_i = base_coord - vec2<i32>(1);
    let end_i = base_coord + vec2<i32>(2);
    let canvas_in_bounds = start_i.x >= 0 && start_i.y >= 0 && end_i.x <= max_coord.x && end_i.y <= max_coord.y;
    let tile_TL = start_i / ts;
    let tile_BR = end_i / ts;
    let is_single_tile = all(tile_TL == tile_BR) && canvas_in_bounds;
    var final_color_linear = vec4<f32>(0.0);
    if (is_single_tile) {
        let idx = tile_TL.y * 2 + tile_TL.x;
        let local_offset = -tile_TL * ts;
        let p_start = start_i + local_offset;
        if (idx == 0) { final_color_linear = catmull_rom_fast1(tex1_0, p_start, wx, wy); }
        else if (idx == 1) { final_color_linear = catmull_rom_fast1(tex1_1, p_start, wx, wy); }
        else if (idx == 2) { final_color_linear = catmull_rom_fast1(tex1_2, p_start, wx, wy); }
        else { final_color_linear = catmull_rom_fast1(tex1_3, p_start, wx, wy); }
    } else {
        final_color_linear = catmull_rom_slow1(start_i, max_coord, wx, wy);
    }
    return clamp(to_srgb_exact(final_color_linear), vec4(0.0), vec4(1.0));
}

fn textureSampleCatmullRom2(uv: vec2<f32>) -> vec4<f32> {
    let tex_size_u = totalDimensions2();
    let tex_size = vec2<f32>(tex_size_u);
    let pixel_coord = uv * tex_size - 0.5;
    let base_coord = vec2<i32>(floor(pixel_coord));
    let f = fract(pixel_coord);
    let wx = catmull_rom_weights(f.x);
    let wy = catmull_rom_weights(f.y);
    let max_coord = vec2<i32>(tex_size_u) - 1;
    let ts = i32(u.tile_size2);
    let start_i = base_coord - vec2<i32>(1);
    let end_i = base_coord + vec2<i32>(2);
    let canvas_in_bounds = start_i.x >= 0 && start_i.y >= 0 && end_i.x <= max_coord.x && end_i.y <= max_coord.y;
    let tile_TL = start_i / ts;
    let tile_BR = end_i / ts;
    let is_single_tile = all(tile_TL == tile_BR) && canvas_in_bounds;
    var final_color_linear = vec4<f32>(0.0);
    if (is_single_tile) {
        let idx = tile_TL.y * 2 + tile_TL.x;
        let local_offset = -tile_TL * ts;
        let p_start = start_i + local_offset;
        if (idx == 0) { final_color_linear = catmull_rom_fast2(tex2_0, p_start, wx, wy); }
        else if (idx == 1) { final_color_linear = catmull_rom_fast2(tex2_1, p_start, wx, wy); }
        else if (idx == 2) { final_color_linear = catmull_rom_fast2(tex2_2, p_start, wx, wy); }
        else { final_color_linear = catmull_rom_fast2(tex2_3, p_start, wx, wy); }
    } else {
        final_color_linear = catmull_rom_slow2(start_i, max_coord, wx, wy);
    }
    return clamp(to_srgb_exact(final_color_linear), vec4(0.0), vec4(1.0));
}

fn loop_over_tile1(tex: texture_2d<f32>, start_i: vec2<i32>, end_i: vec2<i32>, src_start: vec2<f32>, src_end: vec2<f32>, local_offset: vec2<i32>) -> vec4<f32> {
    var color_sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var y: i32 = start_i.y; y < end_i.y; y++) {
        let y_f = f32(y);
        var y_overlap = 1.0;
        if (y == start_i.y) { y_overlap = min(y_f + 1.0, src_end.y) - src_start.y; }
        else if (y == end_i.y - 1) { y_overlap = src_end.y - max(y_f, src_start.y); }
        y_overlap = max(0.0, y_overlap);
        let py = y + local_offset.y;
        for (var x: i32 = start_i.x; x < end_i.x; x++) {
            let x_f = f32(x);
            var x_overlap = 1.0;
            if (x == start_i.x) { x_overlap = min(x_f + 1.0, src_end.x) - src_start.x; }
            else if (x == end_i.x - 1) { x_overlap = src_end.x - max(x_f, src_start.x); }
            x_overlap = max(0.0, x_overlap);
            let weight = x_overlap * y_overlap;
            let px = x + local_offset.x;
            let texel = to_linear_exact(textureLoad(tex, vec2<i32>(px, py), 0));
            color_sum += texel * weight;
            weight_sum += weight;
        }
    }
    return color_sum / max(weight_sum, 0.0001);
}

fn downsample1(src_start: vec2<f32>, scale: vec2<f32>) -> vec4<f32> {
    let src_size_f = vec2<f32>(totalDimensions1());
    let src_end = src_start + scale;
    let start_i = vec2<i32>(clamp(floor(src_start), vec2<f32>(0.0), src_size_f));
    let end_i = vec2<i32>(clamp(ceil(src_end), vec2<f32>(0.0), src_size_f));
    let ts = i32(u.tile_size1);
    let tile_TL = start_i / ts;
    let tile_BR = (end_i - 1) / ts;
    let in_bounds = start_i.x >= 0 && start_i.y >= 0 && (end_i.x - 1) < ts * 2 && (end_i.y - 1) < ts * 2;
    let is_single_tile = all(tile_TL == tile_BR) && in_bounds;
    if (is_single_tile) {
        let idx = tile_TL.y * 2 + tile_TL.x;
        let local_offset = -tile_TL * ts;
        var avg_color = vec4<f32>(0.0);
        if (idx == 0) { avg_color = loop_over_tile1(tex1_0, start_i, end_i, src_start, src_end, local_offset); }
        else if (idx == 1) { avg_color = loop_over_tile1(tex1_1, start_i, end_i, src_start, src_end, local_offset); }
        else if (idx == 2) { avg_color = loop_over_tile1(tex1_2, start_i, end_i, src_start, src_end, local_offset); }
        else { avg_color = loop_over_tile1(tex1_3, start_i, end_i, src_start, src_end, local_offset); }
        return to_srgb_exact(avg_color);
    } else {
        var color_sum = vec4<f32>(0.0);
        var weight_sum = 0.0;
        for (var y: i32 = start_i.y; y < end_i.y; y++) {
            let y_f = f32(y);
            var y_overlap = 1.0;
            if (y == start_i.y) { y_overlap = min(y_f + 1.0, src_end.y) - src_start.y; }
            else if (y == end_i.y - 1) { y_overlap = src_end.y - max(y_f, src_start.y); }
            y_overlap = max(0.0, y_overlap);
            for (var x: i32 = start_i.x; x < end_i.x; x++) {
                let x_f = f32(x);
                var x_overlap = 1.0;
                if (x == start_i.x) { x_overlap = min(x_f + 1.0, src_end.x) - src_start.x; }
                else if (x == end_i.x - 1) { x_overlap = src_end.x - max(x_f, src_start.x); }
                x_overlap = max(0.0, x_overlap);
                let weight = x_overlap * y_overlap;
                let texel = to_linear_exact(totalLoad1(vec2<i32>(x, y)));
                color_sum += texel * weight;
                weight_sum += weight;
            }
        }
        return to_srgb_exact(color_sum / max(weight_sum, 0.0001));
    }
}

fn loop_over_tile2(tex: texture_2d<f32>, start_i: vec2<i32>, end_i: vec2<i32>, src_start: vec2<f32>, src_end: vec2<f32>, local_offset: vec2<i32>) -> vec4<f32> {
    var color_sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var y: i32 = start_i.y; y < end_i.y; y++) {
        let y_f = f32(y);
        var y_overlap = 1.0;
        if (y == start_i.y) { y_overlap = min(y_f + 1.0, src_end.y) - src_start.y; }
        else if (y == end_i.y - 1) { y_overlap = src_end.y - max(y_f, src_start.y); }
        y_overlap = max(0.0, y_overlap);
        let py = y + local_offset.y;
        for (var x: i32 = start_i.x; x < end_i.x; x++) {
            let x_f = f32(x);
            var x_overlap = 1.0;
            if (x == start_i.x) { x_overlap = min(x_f + 1.0, src_end.x) - src_start.x; }
            else if (x == end_i.x - 1) { x_overlap = src_end.x - max(x_f, src_start.x); }
            x_overlap = max(0.0, x_overlap);
            let weight = x_overlap * y_overlap;
            let px = x + local_offset.x;
            let texel = to_linear_exact(textureLoad(tex, vec2<i32>(px, py), 0));
            color_sum += texel * weight;
            weight_sum += weight;
        }
    }
    return color_sum / max(weight_sum, 0.0001);
}

fn downsample2(src_start: vec2<f32>, scale: vec2<f32>) -> vec4<f32> {
    let src_size_f = vec2<f32>(totalDimensions2());
    let src_end = src_start + scale;
    let start_i = vec2<i32>(clamp(floor(src_start), vec2<f32>(0.0), src_size_f));
    let end_i = vec2<i32>(clamp(ceil(src_end), vec2<f32>(0.0), src_size_f));
    let ts = i32(u.tile_size2);
    let tile_TL = start_i / ts;
    let tile_BR = (end_i - 1) / ts;
    let in_bounds = start_i.x >= 0 && start_i.y >= 0 && (end_i.x - 1) < ts * 2 && (end_i.y - 1) < ts * 2;
    let is_single_tile = all(tile_TL == tile_BR) && in_bounds;
    if (is_single_tile) {
        let idx = tile_TL.y * 2 + tile_TL.x;
        let local_offset = -tile_TL * ts;
        var avg_color = vec4<f32>(0.0);
        if (idx == 0) { avg_color = loop_over_tile2(tex2_0, start_i, end_i, src_start, src_end, local_offset); }
        else if (idx == 1) { avg_color = loop_over_tile2(tex2_1, start_i, end_i, src_start, src_end, local_offset); }
        else if (idx == 2) { avg_color = loop_over_tile2(tex2_2, start_i, end_i, src_start, src_end, local_offset); }
        else { avg_color = loop_over_tile2(tex2_3, start_i, end_i, src_start, src_end, local_offset); }
        return to_srgb_exact(avg_color);
    } else {
        var color_sum = vec4<f32>(0.0);
        var weight_sum = 0.0;
        for (var y: i32 = start_i.y; y < end_i.y; y++) {
            let y_f = f32(y);
            var y_overlap = 1.0;
            if (y == start_i.y) { y_overlap = min(y_f + 1.0, src_end.y) - src_start.y; }
            else if (y == end_i.y - 1) { y_overlap = src_end.y - max(y_f, src_start.y); }
            y_overlap = max(0.0, y_overlap);
            for (var x: i32 = start_i.x; x < end_i.x; x++) {
                let x_f = f32(x);
                var x_overlap = 1.0;
                if (x == start_i.x) { x_overlap = min(x_f + 1.0, src_end.x) - src_start.x; }
                else if (x == end_i.x - 1) { x_overlap = src_end.x - max(x_f, src_start.x); }
                x_overlap = max(0.0, x_overlap);
                let weight = x_overlap * y_overlap;
                let texel = to_linear_exact(totalLoad2(vec2<i32>(x, y)));
                color_sum += texel * weight;
                weight_sum += weight;
            }
        }
        return to_srgb_exact(color_sum / max(weight_sum, 0.0001));
    }
}

fn sampleImage1(uv: vec2<f32>, scale_factor: f32) -> vec4<f32> {
    let src_size_f = vec2<f32>(totalDimensions1());
    if (scale_factor > 1.0) {
        let src_start = uv * src_size_f;
        return downsample1(src_start, vec2<f32>(scale_factor));
    } else {
        return textureSampleCatmullRom1(uv);
    }
}

fn sampleImage2(uv: vec2<f32>, scale_factor: f32) -> vec4<f32> {
    let src_size_f = vec2<f32>(totalDimensions2());
    if (scale_factor > 1.0) {
        let src_start = uv * src_size_f;
        return downsample2(src_start, vec2<f32>(scale_factor));
    } else {
        return textureSampleCatmullRom2(uv);
    }
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );

    let uv = uvs[vertex_index];
    let dst_size_f = vec2<f32>(u.dst_width, u.dst_height);
    
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    
    let src_size1 = vec2<f32>(totalDimensions1());
    let src_size2 = vec2<f32>(totalDimensions2());
    
    let screen_pos = uv * dst_size_f;
    let uv1 = (screen_pos / u.scale1 - u.offset1 * dst_size_f) / src_size1;
    let uv2 = (screen_pos / u.scale2 - u.offset2 * dst_size_f) / src_size2;
    
    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv1 = uv1;
    out.uv2 = uv2;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let scale_factor1 = 1.0 / u.scale1;
    let scale_factor2 = 1.0 / u.scale2;
    
    let col1 = sampleImage1(in.uv1, scale_factor1);
    let col2 = sampleImage2(in.uv2, scale_factor2);
    
    let linear1 = to_linear_exact(col1);
    let linear2 = to_linear_exact(col2);
    
    let t = u.blend;
    let blended = mix(linear1, linear2, t);
    
    let result = to_srgb_exact(blended);
    return vec4<f32>(result.rgb * result.a, result.a);
}"""

    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
    ) {
        val image1 = page1.image ?: return
        val image2 = page2.image ?: return
        val res1 = image1.prepareForRender(dst, page1.x, page1.y, page1.scale) ?: return
        val res2 = image2.prepareForRender(dst, page2.x, page2.y, page2.scale) ?: return

        // blend: 0 = fully page1, 1 = fully page2
        val blend = if (frac > 0f) frac else -frac

        val byteBuffer = ByteBuffer.allocateDirect(60).apply {
            order(ByteOrder.nativeOrder())
            // Image 1
            putFloat(0, res1.x)
            putFloat(4, res1.y)
            putFloat(8, res1.scale)
            putFloat(12, res1.mipmap.tilesize.toFloat())
            putFloat(16, res1.mipmap.tilesCols.toFloat())
            putFloat(20, res1.mipmap.tilesRows.toFloat())
            // Image 2
            putFloat(24, res2.x)
            putFloat(28, res2.y)
            putFloat(32, res2.scale)
            putFloat(36, res2.mipmap.tilesize.toFloat())
            putFloat(40, res2.mipmap.tilesCols.toFloat())
            putFloat(44, res2.mipmap.tilesRows.toFloat())
            // Shared
            putFloat(48, dst.width.toFloat())
            putFloat(52, dst.height.toFloat())
            putFloat(56, blend)
        }

        device.queue.writeBuffer(image1.buffer, 0, byteBuffer)

        val pass = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = dst.createView(),
                        loadOp = LoadOp.Load,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                    )
                )
            )
        )

        pass.setPipeline(pipeline)
        pass.setBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = pipeline.getBindGroupLayout(0), entries = arrayOf(
                        GPUBindGroupEntry(0, buffer = image1.buffer),
                    ).plus(res1.quad.tileViews.mapIndexed { i, view ->
                        GPUBindGroupEntry(1 + i, textureView = view)
                    }).plus(res2.quad.tileViews.mapIndexed { i, view ->
                        GPUBindGroupEntry(5 + i, textureView = view)
                    })
                )
            )
        )

        pass.draw(6)
        pass.end()
    }
}
