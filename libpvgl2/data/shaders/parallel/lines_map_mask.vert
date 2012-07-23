/**
 * \file lines_map_mask.vert
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#version 330

uniform mat4 view;
uniform vec2 map_position = vec2(0.0, 0.0);

in vec2 position;
out vec2 pos;

void main (void)
{
  pos = position;
  gl_Position = view * vec4 (position, 0.0, 1.0) + vec4(map_position, 0.0, 0.0);
}

