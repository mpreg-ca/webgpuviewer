package ca.mpreg.webgpuviewer

val TrimShader = """
struct Params {
    background: vec3<f32>,
    threshold: f32,
}

struct TrimResult {
    min_x: atomic<u32>,
    min_y: atomic<u32>,
    max_x: atomic<u32>,
    max_y: atomic<u32>,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> result: TrimResult;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> wg_min_x: atomic<u32>;
var<workgroup> wg_min_y: atomic<u32>;
var<workgroup> wg_max_x: atomic<u32>;
var<workgroup> wg_max_y: atomic<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32
) {
    if (local_invocation_index == 0u) {
        atomicStore(&wg_min_x, 0xFFFFFFFFu);
        atomicStore(&wg_min_y, 0xFFFFFFFFu);
        atomicStore(&wg_max_x, 0u);
        atomicStore(&wg_max_y, 0u);
    }
    workgroupBarrier();

    let dims = vec2<i32>(textureDimensions(input_tex));
    let coords = vec2<i32>(global_id.xy);
    let in_bounds = coords.x < dims.x && coords.y < dims.y;

    if (in_bounds) {
        let color = textureLoad(input_tex, coords, 0);
        let pixel_rgb = color.rgb * color.a + params.background.rgb * (1.0 - color.a);

        let diff = abs(pixel_rgb - params.background.rgb);
        let is_foreground = (diff.r > params.threshold) || 
                            (diff.g > params.threshold) || 
                            (diff.b > params.threshold);

        if (is_foreground) {
            atomicMin(&wg_min_x, u32(coords.x));
            atomicMin(&wg_min_y, u32(coords.y));
            atomicMax(&wg_max_x, u32(coords.x));
            atomicMax(&wg_max_y, u32(coords.y));
        }
    }

    workgroupBarrier();

    if (local_invocation_index == 0u) {
        let w_min_x = atomicLoad(&wg_min_x);
        let w_min_y = atomicLoad(&wg_min_y);
        let w_max_x = atomicLoad(&wg_max_x);
        let w_max_y = atomicLoad(&wg_max_y);

        if (w_min_x != 0xFFFFFFFFu) {
            atomicMin(&result.min_x, w_min_x);
            atomicMin(&result.min_y, w_min_y);
            atomicMax(&result.max_x, w_max_x);
            atomicMax(&result.max_y, w_max_y);
        }
    }
}
"""
val TrimShaderSingle = """
struct Params {
    background: vec3<f32>,
    threshold: f32,
}

struct TrimResult {
    min_x: atomic<u32>,
    min_y: atomic<u32>,
    max_x: atomic<u32>,
    max_y: atomic<u32>,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> result: TrimResult;
@group(0) @binding(2) var<uniform> params: Params;

fn is_foreground(coords: vec2<i32>, dims: vec2<i32>) -> bool {
    let color = textureLoad(input_tex, coords, 0);
    let pixel_rgb = color.rgb * color.a + params.background.rgb * (1.0 - color.a);


    let diff = abs(pixel_rgb - params.background.rgb);
    return (diff.r > params.threshold) || 
           (diff.g > params.threshold) || 
           (diff.b > params.threshold);
}

@compute @workgroup_size(64, 1, 1)
fn find_left(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(input_tex));
    let x = i32(global_id.x); // Thread index represents a column
    if (x >= dims.x) { return; }

    for (var y = 0; y < dims.y; y = y + 1) {
        if (is_foreground(vec2<i32>(x, y), dims)) {
            atomicMin(&result.min_x, u32(x));
            break; // Early exit: first foreground pixel found in this column
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn find_right(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(input_tex));
    let x = i32(global_id.x); // Thread index represents a column
    if (x >= dims.x) { return; }

    for (var y = 0; y < dims.y; y = y + 1) {
        if (is_foreground(vec2<i32>(x, y), dims)) {
            atomicMax(&result.max_x, u32(x));
            break; // Early exit
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn find_top(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(input_tex));
    let y = i32(global_id.x); // Thread index represents a row index
    if (y >= dims.y) { return; }

    for (var x = 0; x < dims.x; x = x + 1) {
        if (is_foreground(vec2<i32>(x, y), dims)) {
            atomicMin(&result.min_y, u32(y));
            break; // Early exit
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn find_bottom(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(input_tex));
    let y = i32(global_id.x); // Thread index represents a row index
    if (y >= dims.y) { return; }

    for (var x = 0; x < dims.x; x = x + 1) {
        if (is_foreground(vec2<i32>(x, y), dims)) {
            atomicMax(&result.max_y, u32(y));
            break; // Early exit
        }
    }
}
"""