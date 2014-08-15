#ifndef COMPUTE_H
#define COMPUTE_H

void initParticles(float** particles, float** velocities, 
		   int numParticles);
void updateParticles(float* particles, float* velocities, int numParticles);

void deleteParticles(float** particles, float** velocities);

void vecNormalize(const float* a, float* b);

#endif//COMPUTE_H
