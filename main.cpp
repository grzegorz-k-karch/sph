#include <iostream>
#include <omp.h>

#include "compute.h"
#include "render.h"

int numParticles = 5000;
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

  initParticles(&particles, &velocities, numParticles);

  int closeWindow = 0;

  while (!closeWindow) {

    clock_t t = clock();

    updateParticles(particles, velocities, numParticles);
    closeWindow = display(particles, numParticles);

    t = clock() - t;
    std::cout << "elapsed time: " << float(t)/CLOCKS_PER_SEC << std::endl;
  }

  if (particles != 0) {
    free(particles);
  }

  terminateOpenGL();

  return 0;
}

