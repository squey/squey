#version 330

out vec4 frag_color;
in  vec2 tex_coord_frag;

uniform sampler2DRect fbo_sampler;

void main(void)
{
#if 0
  if ((int(gl_FragCoord.x + gl_FragCoord.y) % 2) == 0)
   discard;
  frag_color =  texture (fbo_sampler, tex_coord_frag);
#else
#if 0
  frag_color = (texture (fbo_sampler, tex_coord_frag + vec2(-1, -1)) +
                texture (fbo_sampler, tex_coord_frag + vec2(-1,  0)) +
                texture (fbo_sampler, tex_coord_frag + vec2(-1,  1)) +
                texture (fbo_sampler, tex_coord_frag + vec2( 0, -1)) +
                texture (fbo_sampler, tex_coord_frag + vec2( 0,  0)) +
                texture (fbo_sampler, tex_coord_frag + vec2( 0,  1)) +
                texture (fbo_sampler, tex_coord_frag + vec2( 1, -1)) +
                texture (fbo_sampler, tex_coord_frag + vec2( 1,  0)) +
                texture (fbo_sampler, tex_coord_frag + vec2( 1,  1))) / 18.0;
#else
  frag_color = texture (fbo_sampler, tex_coord_frag) / 2.0;
#endif
#endif
}

