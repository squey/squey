/**
 * \file lines_zombie_fbo.frag
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#version 330

out vec4 frag_color;
in  vec2 tex_coord_frag;

uniform sampler2DRect fbo_sampler;

uniform bool draw_unselected = true;
uniform bool draw_zombie = true;

void main(void)
{
  vec4 color = texture(fbo_sampler, tex_coord_frag);
    if (color.rgb != vec3(0.0 ,0.0 ,0.0) && draw_unselected) { // this fragment is selected.
      frag_color = vec4(color.rgb, 1.0);
    } else if (color.a > 0.5 && draw_zombie) { // this fragment is zombie
      frag_color = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
      frag_color = vec4(0.2, 0.2, 0.2, 1.0);
    }
}

