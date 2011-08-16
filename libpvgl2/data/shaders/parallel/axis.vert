#version 330

in vec3 position;
in vec4 color;
uniform mat4 modelview;

out float y;
out vec4 v_color;



void main(void)
{
  y = position.y;
  gl_Position = modelview * vec4 (position, 1.0);
  v_color = color;
}
