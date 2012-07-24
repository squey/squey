/**
 * \file dot.vert
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#version 330

uniform mat4 view;
//uniform int   eventline_first = 0;
//uniform int   eventline_current = 1;
//uniform usamplerBuffer selection_sampler;

in  vec4 color_v;
out vec4 color;
out float v_id;

in float zla_v;
in vec2 position;

void main (void)
{
/*  if (gl_PrimitiveIDIn+drawn_lines > eventline_current || gl_PrimitiveIDIn +drawn_lines< eventline_first)
    return;
    int chunk_offset = (gl_PrimitiveIDIn+drawn_lines) / 32;
    uint selection_chunk = texelFetch(selection_sampler, chunk_offset).r;
    if ((selection_chunk & (1u << (uint(gl_PrimitiveIDIn+drawn_lines) & 31u))) == 0u)
    return;
    mat4 view2 = mat4(vec4(1.0/zoom.x,0.0,0.0,0.0),
    vec4(0.0,1.0/zoom.y,0.0,0.0),
    vec4(0.0,0.0,1.0,0.0),
    vec4(0.0,0.0,0.0,1.0));*/
color = color_v;
v_id = gl_VertexID;
gl_Position = view * vec4 (position, zla_v, 1.0);
}
