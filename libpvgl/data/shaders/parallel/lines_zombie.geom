

precision highp float;

layout(points) in;
layout(line_strip, max_vertices = coordinates) out;

#extension GL_EXT_geometry_shader4 : enable

uniform mat4 view;
uniform vec2 zoom = vec2(1.0, 1.0);
uniform float batch;
uniform usamplerBuffer zombie_sampler;
uniform int drawn_lines = 0;

in  vec3 color_g[];
out vec4 color;

in vec4 position_g[][tab_size];

void main (void)
{
  int chunk_offset = (gl_PrimitiveIDIn + drawn_lines) / 32;
  uint selection_chunk = texelFetch(zombie_sampler, chunk_offset).r;
  if ((selection_chunk & (1u << (uint(gl_PrimitiveIDIn + drawn_lines) & 31u))) != 0u)
    color = vec4(color_g[0], 0.0);
  else
    color = vec4(0.0, 0.0, 0.0, 1.0);
	mat4 view2 = mat4(vec4(1.0/zoom.x,0.0,0.0,0.0),
										vec4(0.0,1.0/zoom.y,0.0,0.0),
										vec4(0.0,0.0,1.0,0.0),
										vec4(0.0,0.0,0.0,1.0));
for (int i = 0; i < coordinates / 4; i++)
  {
  gl_Position = view2 * view * vec4 (55.0 * batch + 4.0 * float(i) + 0.0, position_g[0][i].x, 0.0, 1.0);
  EmitVertex();
  gl_Position = view2 * view * vec4 (55.0 * batch + 4.0 * float(i) + 1.0, position_g[0][i].y, 0.0, 1.0);
  EmitVertex();
  gl_Position = view2 * view * vec4 (55.0 * batch + 4.0 * float(i) + 2.0, position_g[0][i].z, 0.0, 1.0);
  EmitVertex();
  gl_Position = view2 * view * vec4 (55.0 * batch + 4.0 * float(i) + 3.0, position_g[0][i].w, 0.0, 1.0);
  EmitVertex();
  }
for (int i = 0; i < coordinates % 4; i++)
  {
  gl_Position = view2 * view * vec4 (55.0 * batch + float(4 * (tab_size-1) + i), position_g[0][tab_size-1][i], 0.0, 1.0);
  EmitVertex();
  }
EndPrimitive();
}
