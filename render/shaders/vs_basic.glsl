#version 440

uniform mat4 proj_mat;
uniform mat4 model_mat;
uniform mat4 view_mat;
uniform mat4 model_it_mat;

uniform vec3 domainScale;
uniform vec3 domainOffset;

layout (location = 0) in vec4 in_position;
layout (location = 1) in vec4 in_normal;
layout (location = 2) in vec3 in_center;
layout (location = 2) in vec3 in_velo;

layout (location = 0) out vec3 out_color;
layout (location = 1) out vec3 out_L;
layout (location = 2) out vec3 out_N;

void main () 
{
  vec4 domainPos = vec4(domainScale*vec3(in_position.xyz+in_center)+domainOffset, 1.0);
  vec4 worldPos = model_mat*domainPos;
  gl_Position = proj_mat*view_mat*worldPos;

  //out_color = normalize(vec3(float(gl_InstanceID)/2000.0,1.0,1.0));
  out_color = vec3(0.2, 0.6, 0.9);

  vec3 lightPos = vec3(10.0, 10.0, 10.0);

  out_N = normalize((model_it_mat*vec4(in_normal.xyz, 0.0)).xyz);
  out_L = normalize(lightPos - worldPos.xyz);
}
