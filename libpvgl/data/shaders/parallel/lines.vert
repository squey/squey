
in  vec4 color_v;
out vec4 color_g;
in  vec4 zla_v;
out vec4 zla_g;

in  vec4 position_v[tab_size];
out vec4 position_g[tab_size];

void main (void)
{
  zla_g = zla_v;
  color_g = color_v;
  for (int i = 0; i < tab_size ; i++)
    {
      position_g[i] = position_v[i];
    }
}
