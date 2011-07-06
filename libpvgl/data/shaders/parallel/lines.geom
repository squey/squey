

precision highp float;

layout(points) in;
layout(line_strip, max_vertices = coordinates) out;

#extension GL_EXT_geometry_shader4 : enable

uniform mat4 view;
uniform vec2 zoom = vec2(1.0, 1.0);
uniform float batch;
uniform int   eventline_first = 0;
uniform int   eventline_current = 1;
uniform int   drawn_lines = 0;

uniform usamplerBuffer selection_sampler;

in  vec4 color_g[];
out vec4 color;

in vec4 zla_g[];
in vec4 position_g[][tab_size];

void main (void)
{
  if (gl_PrimitiveIDIn+drawn_lines > eventline_current || gl_PrimitiveIDIn +drawn_lines< eventline_first)
    return;
  int chunk_offset = (gl_PrimitiveIDIn+drawn_lines) / 32;
  uint selection_chunk = texelFetch(selection_sampler, chunk_offset).r;
  if ((selection_chunk & (1u << (uint(gl_PrimitiveIDIn+drawn_lines) & 31u))) == 0u)
    return;
	mat4 view2 = mat4(vec4(1.0/zoom.x,0.0,0.0,0.0),
										vec4(0.0,1.0/zoom.y,0.0,0.0),
										vec4(0.0,0.0,1.0,0.0),
										vec4(0.0,0.0,0.0,1.0));
color = color_g[0];
for (int i = 0; i < coordinates / 4; i++)
  {
  gl_Position = view2 * view * vec4 (55.0 * batch + 4.0 * float(i) + 0.0, position_g[0][i].x, zla_g[0].x, 1.0);
  EmitVertex();
  gl_Position = view2 * view * vec4 (55.0 * batch + 4.0 * float(i) + 1.0, position_g[0][i].y, zla_g[0].x, 1.0);
  EmitVertex();
  gl_Position = view2 * view * vec4 (55.0 * batch + 4.0 * float(i) + 2.0, position_g[0][i].z, zla_g[0].x, 1.0);
  EmitVertex();
  gl_Position = view2 * view * vec4 (55.0 * batch + 4.0 * float(i) + 3.0, position_g[0][i].w, zla_g[0].x, 1.0);
  EmitVertex();
  }
for (int i = 0; i < coordinates % 4; i++)
  {
  gl_Position = view2 * view * vec4 (55.0 * batch + float(4 * (tab_size-1) + i), position_g[0][tab_size-1][i], zla_g[0].x, 1.0);
  EmitVertex();
  }
EndPrimitive();
}
