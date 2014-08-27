#ifndef RENDER_H
#define RENDER_H

void initOpenGL(float tankSize);

void initGeometry();

int display(float* particles, int numParticles);

void terminateOpenGL();

#endif//RENDER_H
