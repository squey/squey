#version 330

uniform int   eventline_first = 0;
uniform int   eventline_current = 1;
uniform usamplerBuffer selection_sampler;
uniform usamplerBuffer zombie_sampler;
uniform float draw_unselected = 1.0;
uniform float draw_zombie = 1.0;

in vec4 color;
out vec4 frag_color;
in float v_id;

void main(void)
{
  int chunk_offset = int(v_id) / 32;
  uint zombie_chunk = texelFetch(zombie_sampler, chunk_offset).r;
  if ((zombie_chunk & (1u << (uint(v_id) & 31u))) == 0u) {
      if (draw_zombie > 0.5)
          frag_color = vec4(0.0, 0.0, 0.0, 1.0);
      else
          discard;
  } else {
      uint selection_chunk = texelFetch(selection_sampler, chunk_offset).r;
      if ((selection_chunk & (1u << (uint(v_id) & 31u))) == 0u || v_id > eventline_current || v_id < eventline_first) {
          if (draw_unselected > 0.5)
              frag_color = vec4(color.rgb / 2.0, 1.0);
          else
              discard;
      } else
          frag_color = color;
  }
}

