/**
 * \file lines_zombie.vert
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

in  vec4 color_v;
out vec3 color_g;
in  vec4 position_v[tab_size];
out vec4 position_g[tab_size];

void main (void)
{
  color_g = color_v.rgb / 2.0;
  for (int i = 0; i < tab_size ; i++)
    {
      position_g[i] = position_v[i];
    }
}
