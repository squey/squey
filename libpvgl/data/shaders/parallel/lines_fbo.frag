#version 330

out vec4 frag_color;
in  vec2 tex_coord_frag;

uniform sampler2DRect fbo_sampler;

void main(void)
{
  frag_color = texture (fbo_sampler, tex_coord_frag);
}

