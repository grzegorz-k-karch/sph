#version 440

layout (location = 0) in vec3 in_color;
layout (location = 1) in vec3 in_L;
layout (location = 2) in vec3 in_N;

out vec4 out_color;

void main () 
{
  out_color = vec4(in_color, 1.0);

  vec3 NN = normalize(in_N);
  vec3 NL = normalize(in_L);
  vec3 NH = normalize(NL+vec3(0.0, 0.0, 1.0));

  float ka = 0.5; 
  float kd = 0.45*max(0.0, dot(NL, NN));
  const float specularExp = 128.0;
  float ks = 0.125*pow(max(0.0,dot(NN,NH)),specularExp);

  out_color = vec4(vec3(kd + ka)*in_color + vec3(ks), 1.0);
}
