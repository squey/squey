/**
 * \file lines_unselected.frag
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#version 330

in vec4 color;
out vec4 frag_color;

void main(void)
{
  frag_color = color;
//  frag_color = vec4(0.0, 1.0, 0.0, 1.0);
}
