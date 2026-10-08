#version 440

// One vertical span per sample, exactly as PVSeriesRendererRaster draws it on the CPU:
// the subsampler yields one value per column of pixels, so a serie is a run of spans
// rather than a polyline, and both backends then produce the same picture. The geometry
// is derived from the vertex index alone, so nothing but the sampled rows ever travels
// to the GPU.

layout(location = 0) out vec4 v_color;

layout(std140, binding = 0) uniform buf {
    int samples_count;
    int width;
    int height;
    int points_mode;
    // -1 where the clip space has Y pointing up, 1 where it points down: Vulkan differs
    // from the rest and QRhi leaves the correction to whoever writes the shader.
    float y_sign;
    float padding[3];
} ubuf;

// Row of every sample of every serie, offset by two so that zero can mean "no value" and
// the row just above the top edge stays positive. The two bytes are carried by the two
// channels of an RG8 texel.
layout(binding = 1) uniform sampler2D rows_tex;
layout(binding = 2) uniform sampler2D colors_tex;

const int row_bias = 2;

int encoded_row_at(int x, int serie)
{
    vec4 texel = texelFetch(rows_tex, ivec2(x, serie), 0);
    int lo = int(texel.r * 255.0 + 0.5);
    int hi = int(texel.g * 255.0 + 0.5);
    return lo | (hi << 8);
}

void degenerate()
{
    // Outside the clip volume, so the whole primitive is thrown away.
    gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
    v_color = vec4(0.0);
}

void main()
{
    int sample_index = gl_VertexIndex / 2;
    int span_end = gl_VertexIndex - sample_index * 2;
    int serie = sample_index / ubuf.samples_count;
    int x = sample_index - serie * ubuf.samples_count;

    int from = encoded_row_at(x, serie);
    if (from == 0) {
        degenerate();
        return;
    }
    // The sample of the next column closes the span; the last one, and any sample the
    // next column does not join, stands alone.
    int to = from;
    if (ubuf.points_mode == 0 && x + 1 < ubuf.samples_count) {
        int next = encoded_row_at(x + 1, serie);
        if (next != 0) {
            to = next;
        }
    }

    int top = max(min(from, to) - row_bias, 0);
    int bottom = min(max(from, to) - row_bias, ubuf.height - 1);
    if (top > bottom) {
        degenerate(); // entirely above or below the plot
        return;
    }

    // From the centre of the first row to the centre of the one past the last: the last
    // pixel of a line is not drawn, so this lights exactly [top, bottom].
    float row = span_end == 0 ? float(top) : float(bottom + 1);
    gl_Position = vec4((float(x) + 0.5) / float(ubuf.width) * 2.0 - 1.0,
                       ((row + 0.5) / float(ubuf.height) * 2.0 - 1.0) * ubuf.y_sign, 0.0, 1.0);
    v_color = texelFetch(colors_tex, ivec2(serie, 0), 0);
}
