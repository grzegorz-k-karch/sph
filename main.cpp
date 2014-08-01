#include <iostream>

#include <omp.h>

#include "compute.h"
#include "render.h"

int numParticles = 3000;
float *particles = 0;
float *velocities = 0;

extern float tankSize;

int omp_thread_count() 
{
  int n = 0;
#pragma omp parallel reduction(+:n)
  n += 1;
  return n;
}

int main()
{
  initOpenGL();
  initGeometry();

  float scale[3] = {tankSize/2.0f, tankSize/2.0f, tankSize/2.0f};
  float offset[3] = {0.0f, 0.0f, 0.0f};

  initParticles(&particles, &velocities, numParticles, scale, offset);

  int closeWindow = 0;

  while (!closeWindow) {

    updateParticles(particles, velocities, numParticles);
    closeWindow = display(particles, numParticles);
  }

  if (particles != 0) {
    free(particles);
  }

  terminateOpenGL();

  return 0;
}

