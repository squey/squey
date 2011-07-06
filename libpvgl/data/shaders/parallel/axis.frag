#version 330

uniform float time;

uniform int axis_mode = 0;

out vec4  frag_color;
in  float y;
in  vec4  v_color;

void main(void)
{
  vec4 color1 = vec4 (1.0, 1.0, 1.0, 1.0);
  vec4 color2 = vec4 (0.0, 1.0, 1.0, 1.0);
  float value = time / 1000.0 + y;
  value = value - floor(value);
  vec4 color = mix(color1, color2, value);
  if (axis_mode != 0)
    frag_color = color * value;
  else
    frag_color = v_color;
}

