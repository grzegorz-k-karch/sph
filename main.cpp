#include <iostream>
#include <omp.h>
#include <sys/time.h>

#include "compute.h"
#include "render.h"

int numParticles = 5000;
float *particles = 0;
float *velocities = 0;

extern float tankSize;

static int omp_thread_count() 
{
  int n = 0;
#pragma omp parallel reduction(+:n)
  n += 1;
  return n;
}

int main()
{
  std::cout << "NTHREADS = " << omp_thread_count() << std::endl;

  initOpenGL();
  initGeometry();

  initParticles(&particles, &velocities, numParticles);

  int closeWindow = 0;

  struct timeval start;
  struct timeval end;

  while (!closeWindow) {

    // clock_t t = clock();

    gettimeofday(&start, 0);

    updateParticles(particles, velocities, numParticles);
    closeWindow = display(particles, numParticles);

    // t = clock() - t;
    // std::cout << "elapsed time: " << float(t)/CLOCKS_PER_SEC << std::endl;

    gettimeofday(&end, 0);
    time_t useconds = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    std::cout << "elapsed time [us]: " << useconds << std::endl;
  }

  if (particles != 0) {
    free(particles);
  }

  terminateOpenGL();

  return 0;
}

