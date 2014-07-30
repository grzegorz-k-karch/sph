#ifndef COMPUTE_H
#define COMPUTE_H

void initParticles(float** particles, float** velocities, 
		   int numParticles, float scale[3], float offset[3]);
void updateParticles(float* particles, float* velocities, int numParticles);

void vecNormalize(const float* a, float* b);

#endif//COMPUTE_H
