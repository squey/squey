/**
 * \file lines_map_fbo.vert
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#version 330

uniform vec2 size = vec2(500.0, 500.0);
uniform vec2 size_map = vec2(500.0 / 3.0, 500.0 / 3.0);

in vec2 position;
in vec2 tex_coord;
out vec2 tex_coord_frag;

void main (void)
{
  gl_Position = vec4 (position.x / size.x * 2.0 - 1.0,
										  position.y / size.y * 2.0 - 1.0, 0.0, 1.0);
  tex_coord_frag = tex_coord * size_map;
}

