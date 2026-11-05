// Identity: one pass that copies the tile.
//
// Read by ImageFilterWgsl. The format is mpv's user shader headers, with WGSL compute shaders
// (entry point `main`) for the bodies. A `//!` line after a body, but for //!PARAM, starts the next
// pass.
//
// As in mpv:
//   //!DESC text       the pass's name
//   //!HOOK point      MAIN or RGB: the tile, between being drawn and being resized. Optional,
//                      as that is the only place a pass runs
//   //!BIND name       texture for the pass's next `texture_2d` binding, in binding order.
//                      HOOKED is the tile; any other name is an earlier pass's //!SAVE
//   //!SAVE name       what the pass's `texture_storage_2d` binding writes. Without it the pass
//                      replaces HOOKED, so the last pass has none: that is the network's result.
//                      Storage format comes from the WGSL declaration. Saving a name again
//                      replaces it for later passes
//   //!WIDTH expr      size of what the pass writes, by default HOOKED.w and HOOKED.h
//   //!HEIGHT expr
//   //!WHEN expr       the pass runs only for tiles where this is not 0. As an image filter, whose
//                      output serves every zoom, it is asked once, as at 100%
//   //!COMPONENTS n    checked to be 1 to 4, otherwise unused
//   //!COMPUTE bw bh [tw th]
//                      output pixels per workgroup, by default @workgroup_size, which also sets
//                      the threads: tw th are ignored. Only compute passes are supported
// The expressions are mpv's reverse Polish: numbers, NAME.w and NAME.h (HOOKED, OUTPUT, or an
// earlier //!SAVE), ! + - * / % > < >= <= == != && || and the ternary ?. OUTPUT is the size the
// tile will be resized to, so OUTPUT.w HOOKED.w / is the tile's page scale. In //!WHEN only,
// those are for a nominal 256-pixel tile: ratios mean something, absolute sizes don't.
//
// Also as in mpv, a block of its own, ended by the first line not its own - best just above the
// override below, within the pass, where it stays a comment of the WGSL:
//   //!PARAM name      a value the app sets at run time (ImageFilterWgsl.set), at most 64
//   //!DESC text
//   //!TYPE type       float (the default), int or uint
//   //!DEFAULT n       required
//   //!MINIMUM n       what ImageFilterWgsl.set clamps to; unbounded without
//   //!MAXIMUM n
// Unlike mpv, a pass reads it by declaring a WGSL `override` of that name, of the matching type
// f32, i32 or u32, without @id; any initializer is not used. A mismatched type fails the filter if
// a value doesn't convert. A new value rebuilds the pipelines of the passes that declare it. Other
// overrides are left alone.
//
// Not in mpv:
//   //!HALO expr       input pixels of surrounding page an output pixel reads, since a tile is not
//                      a whole frame. Before any pass it is the default; in a header, that pass's.
//                      The network needs the most of the passes that run. An expression as above,
//                      but of numbers and //!PARAM names, rounded up: 0 to 256
//
// Also not in mpv: a pass declaring `var<immediate> name: HookInfo;` reads where the filter's
// input lies on its page. The struct is added to every pass, so none may define its own, and a
// pass may declare no other var<immediate>:
//   origin: vec2<f32>      the input's texel (0, 0) top-left, from the page's
//   page_size: vec2<f32>   the page's size
//   page_scale: f32        the scale the filter runs at
//   input_scale: f32       input pixels per page pixel
//   output_size: f32       the side, in pixels, the chain's result is cut to: the tile on screen
//   page_seed: u32         fixed per page, for noise that matches across its tiles
// origin and page_size are in the network's input pixels: a pass reading a resized texture
// scales them itself. As an image filter, the page is the image, page_scale is 1, output_size is
// the block as kept in the image's level, and page_seed is 0, as that serves every page.
//
// A network has no resolve of its own: the tile chain resizes its output.
//
// A `sampler` binding gets a linear clamp sampler. Textures are reused once their last reader
// has run.

//!DESC identity
//!HOOK MAIN
//!BIND HOOKED
//!COMPUTE 8 8
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= textureDimensions(outputTexture))) {
        return;
    }
    textureStore(outputTexture, id.xy, textureLoad(tex, id.xy, 0));
}
