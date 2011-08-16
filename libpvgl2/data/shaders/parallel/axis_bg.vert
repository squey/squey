#version 330

in vec3 position;

out float y;
uniform mat4 modelview;

void main(void)
{
  gl_Position = modelview * vec4(position, 1.0);
  y = position.y;
}
