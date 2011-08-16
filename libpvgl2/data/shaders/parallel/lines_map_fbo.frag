#version 330

out vec4 frag_color;
in  vec2 tex_coord_frag;

uniform sampler2DRect map_fbo_sampler;

void main(void)
{
  frag_color = texture (map_fbo_sampler, tex_coord_frag);
}

