#ifndef COMPUTE_H
#define COMPUTE_H

#include <vector>
#include <vector_types.h>

void initParticles(std::vector<float4>& particles, 
		   std::vector<float4>& velocities, 
		   unsigned numParticles);
void updateParticles(std::vector<float4>& particles, 
		     std::vector<float4>& velocities);

#endif//COMPUTE_H
