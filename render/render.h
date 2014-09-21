#ifndef RENDER_H
#define RENDER_H

#include <vector>
#include <vector_types.h>

void initOpenGL(float tankSize);

void initGeometry();

int display(std::vector<float4>& particles);

void terminateOpenGL();

#endif//RENDER_H
