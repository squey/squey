#version 330

uniform vec2 min_mask;
uniform vec2 max_mask;

out vec4 frag_color;
in  vec2 pos;

void main(void)
{
  if (pos.x > min_mask.x &&
      pos.y > min_mask.y &&
      pos.x < max_mask.x &&
      pos.y < max_mask.y)
    discard;
  else
    frag_color = vec4(0.1, 0.1, 0.1, 0.6);
}

