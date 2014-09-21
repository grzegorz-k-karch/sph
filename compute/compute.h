#ifndef COMPUTE_H
#define COMPUTE_H

void initParticles(float** particles, float** velocities, unsigned numParticles);
void updateParticles(float* particles, float* velocities, unsigned numParticles);

void deleteParticles(float** particles, float** velocities);

void vecNormalize(const float* a, float* b);

#endif//COMPUTE_H
