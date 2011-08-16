#version 330

uniform mat4 modelview;

in vec3 position;

void main(void)
{
  gl_Position = modelview * vec4(position, 1.0);
}
