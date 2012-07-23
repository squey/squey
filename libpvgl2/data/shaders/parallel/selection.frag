/**
 * \file selection.frag
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#version 330

uniform vec4 color = vec4(255.0, 100.0, 30.0, 255.0) / 255.0;

out vec4 frag_color;

void main(void)
{
  frag_color = color;
}

