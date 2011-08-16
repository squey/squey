#version 330

uniform vec4 color = vec4 (130.0, 100.0, 25.0, 255.0) / 255.0;

out vec4 frag_color;

in float y;

void main(void)
{
  if ((int(gl_FragCoord.x+gl_FragCoord.y) % 2) == 1)
    discard;
  frag_color = mix(vec4(0.0, 1.0, 0.0, 1.0), vec4(0.0, 0.0, 1.0, 1.0), y);
}

