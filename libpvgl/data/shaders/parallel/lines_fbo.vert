/**
 * \file lines_fbo.vert
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#version 330

uniform vec2 size = vec2(500.0, 500.0);
uniform vec2 offset = vec2(0.0, 0.0);
uniform vec2 zoom = vec2(1.0, 1.0);

in vec2 position;
in vec2 tex_coord;
out vec2 tex_coord_frag;

void main (void)
{
  gl_Position = vec4 (zoom*(position + 2.0 * offset / size), 0.0, 1.0);
  tex_coord_frag = tex_coord * size;
}

