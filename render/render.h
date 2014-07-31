#ifndef RENDER_H
#define RENDER_H

void initOpenGL();

void initGeometry();

int display(float* particles, int numParticles);

void terminateOpenGL();

#endif//RENDER_H
