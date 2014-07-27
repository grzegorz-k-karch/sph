#version 440

uniform mat4 proj_mat;
uniform mat4 model_mat;
uniform mat4 view_mat;

in vec3 vp;

layout (location = 0) out vec3 out_color;

void main () 
{
  gl_Position = proj_mat*view_mat*model_mat*vec4(vp, 1.0);
  out_color = vec3(0.0);
}
