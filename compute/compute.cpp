#include <time.h>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector> 
#include <forward_list>
#include <algorithm>

#include "compute.h"

#define USE_GRID 1
#define SINGLE_PASS 1

float soundSpeed = 1.0f;
float m = 0.00020543f;
float rho0 = 600.0f;
float dynVisc = 0.2f;
float a_gravity[3] = {0.0f, -10.0f, 0.0f};
float timestep = 0.005f;
float tankSize = 0.2154f;
float surfTension = 0.0378f;//0.0728f;

float h = 0.01f;
float h9 = 0.0f;
float h2 = 0.0f;
float h6 = 0.0f;

#if USE_GRID
std::vector<std::forward_list<int> > plists;
#endif

float vecLength(const float* a)
{
  return sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
}

void vecNormalize(const float* a, float* b)
{
  float len = vecLength(a);
  if (len > 0.0f) {
    b[0] = a[0]/len;
    b[1] = a[1]/len;
    b[2] = a[2]/len;
  }
  else {
    b[0] = b[1] = b[2] = 0.0f;
  }
}

float laplWviscosity(const float r)
{
  float w = 0.0f;
  if (r <= h) {
    w = 45.0f/(M_PI*h6)*(h-r);
  }
  return w;
}

float gradWspiky(const float r)
{
  float w = 0.0f;
  if (r <= h) {
    float hr = h - r;
    float hr2 = hr*hr;
    w = -45.0f/(M_PI*h6)*hr2;
  }
  return w;
}

float Wpoly6(const float r)
{
  float w = 0.0f;
  if (r <= h) {
    float r2 = r*r;
    float hr3 = h2 - r2;
    hr3 = hr3*hr3*hr3;
    w = 315.0f/(64.0f*M_PI*h9)*hr3;
  }
  return w;
}

float gradWpoly6(const float r)
{
  float w = 0.0f;
  if (r <= h) {
    float r2 = r*r;
    float hr2 = h2 - r2;
    hr2 = hr2*hr2;
    w = -945.0f/(32.0f*M_PI*h9)*r*hr2;
  }
  return w;
}

float laplWpoly6(const float r)
{
  float w = 0.0f;
  if (r <= h) {
    float r2 = r*r;
    float hr = h2 - r2;
    float hr2 = hr*hr;
    w = 945.0f/(32.0f*M_PI*h9)*(4.0f*r2*hr - hr2);
  }
  return w;
}

void initParticles(float** particles, float** velocities, 
		   int numParticles)
{
  h2 = h*h;
  h6 = h2*h2*h2;
  h9 = h*h2*h6;

  *particles = new float[numParticles*3];
  *velocities = new float[numParticles*3];
  srand(0);

  for (int i = 0; i < numParticles; i++) {

    (*particles)[i*3+0] = float(rand())/RAND_MAX*tankSize/2.0f;
    (*particles)[i*3+1] = float(rand())/RAND_MAX*tankSize/2.0f;
    (*particles)[i*3+2] = float(rand())/RAND_MAX*tankSize/2.0f;

    (*velocities)[i*3+0] = 0.0f;
    (*velocities)[i*3+1] = 0.0f;
    (*velocities)[i*3+2] = 0.0f;
  }
}
#if USE_GRID
void assignParticlesToCells(float* particles, int numParticles)
{
  const float cellSize = h;
  const int gridSize = std::ceil(tankSize/cellSize);
  const int numCells = gridSize*gridSize*gridSize;

  plists.clear();
  plists.resize(numCells);

  for (int i = 0; i < numParticles; i++) {

    float *pos = &particles[i*3];
    int cellCoords[3] = {int(pos[0]/cellSize), 
			 int(pos[1]/cellSize), 
			 int(pos[2]/cellSize)};
    int cellId = 
      cellCoords[0] + 
      cellCoords[1]*gridSize + 
      cellCoords[2]*gridSize*gridSize;

    plists[cellId].push_front(i);
  }      
}
#endif

void updateParticles(float* particles, float* velocities, int numParticles)
{
#if USE_GRID
  assignParticlesToCells(particles, numParticles);
#endif

  float *f_total = (float*)malloc(numParticles*3*sizeof(float));
  for (int i = 0; i < numParticles*3; i++) {
    f_total[i] = 0.0f;
  }

  float avgDensity = 0.0f;
  float *densities = (float*)malloc(numParticles*sizeof(float));

  // compute density for each particle-----------------------------------------
#pragma omp parallel for 
  for (int i = 0; i < numParticles; i++) {
    densities[i] = 0.0f;
  }

  float own_rho = m*Wpoly6(0.0f);
#if USE_GRID
  const float cellSize = h;
  const int gridSize = std::ceil(tankSize/cellSize);
#endif

  for (int i = 0; i < numParticles; i++) {

    float *ri = &particles[i*3];
#if USE_GRID

#if SINGLE_PASS
    densities[i] += own_rho;
#endif
    int cellCoords[3] = {int(ri[0]/cellSize), 
    			 int(ri[1]/cellSize), 
    			 int(ri[2]/cellSize)};

    int xext[2] = {cellCoords[0]-1, cellCoords[0]+1};
    if (xext[0] < 0) xext[0] = 0;
    if (xext[1] >= gridSize) xext[1] = gridSize-1;

    int yext[2] = {cellCoords[1]-1, cellCoords[1]+1};
    if (yext[0] < 0) yext[0] = 0;
    if (yext[1] >= gridSize) yext[1] = gridSize-1;

    int zext[2] = {cellCoords[2]-1, cellCoords[2]+1};
    if (zext[0] < 0) zext[0] = 0;
    if (zext[1] >= gridSize) zext[1] = gridSize-1;
	  
    for (int z = zext[0]; z <= zext[1]; z++) {
      for (int y = yext[0]; y <= yext[1]; y++) {
	for (int x = xext[0]; x <= xext[1]; x++) {
	  
	  int cellId = x + y*gridSize + z*gridSize*gridSize;

	  if (plists[cellId].empty()) {
	    continue;
	  }

	  for (int& j : plists[cellId]) {
#if SINGLE_PASS
	    if (j <= i) {
	      continue;
	    }
#endif
	    float *rj = &particles[j*3];
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
	    float magr = std::sqrt(r2);
	    float rho = m*Wpoly6(magr);

	    densities[i] += rho;
#if SINGLE_PASS
	    densities[j] += rho; 
#endif
	  }
    	}
      }
    }
#else
    densities[i] += own_rho;
    for (int j = i+1; j < numParticles; j++) {

      float *rj = &particles[j*3];
      float r[3] = {(ri[0]-rj[0]), 
		    (ri[1]-rj[1]), 
		    (ri[2]-rj[2])};
      float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
      float magr = std::sqrt(r2);
      float rho = m*Wpoly6(magr);

      densities[i] += rho;
      densities[j] += rho; 
    }
#endif
  }

  // static int testStep = 1;
  // static int done = 0;

  // if (done == testStep) {
  //   std::cout << "densities[0] = " << densities[0] << std::endl;
  // }
  // done++;

  // pressure-----------------------------------------------------------------
  float c = soundSpeed*soundSpeed;
  for (int i = 0; i < numParticles; i++) {

    float *ri = &particles[i*3];
    float pi = c*(densities[i]-rho0);

#if USE_GRID
    int cellCoords[3] = {int(ri[0]/cellSize), 
    			 int(ri[1]/cellSize), 
    			 int(ri[2]/cellSize)};

    int xext[2] = {cellCoords[0]-1, cellCoords[0]+1};
    if (xext[0] < 0) xext[0] = 0;
    if (xext[1] >= gridSize) xext[1] = gridSize-1;

    int yext[2] = {cellCoords[1]-1, cellCoords[1]+1};
    if (yext[0] < 0) yext[0] = 0;
    if (yext[1] >= gridSize) yext[1] = gridSize-1;

    int zext[2] = {cellCoords[2]-1, cellCoords[2]+1};
    if (zext[0] < 0) zext[0] = 0;
    if (zext[1] >= gridSize) zext[1] = gridSize-1;
	  
    for (int z = zext[0]; z <= zext[1]; z++) {
      for (int y = yext[0]; y <= yext[1]; y++) {
	for (int x = xext[0]; x <= xext[1]; x++) {

	  int cellId = x + y*gridSize + z*gridSize*gridSize;

	  if (plists[cellId].empty()) {
	    continue;
	  }

	  for (int& j : plists[cellId]) {

#if SINGLE_PASS
	    if (j <= i) {
	      continue;
	    }
#endif
	    float *rj = &particles[j*3];
	    float pj = c*(densities[j]-rho0);
      
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
	    float mfp = -m*(pi+pj)/(densities[i]+densities[j])*gradWspiky(magr);
	    // FROM FLUIDS: (does not work well)
	    // float mfp = -m*(pi+pj)/(densities[i]*densities[j])*gradWspiky(magr);

	    vecNormalize(r,r);

	    f_total[i*3+0] += mfp*r[0];
	    f_total[i*3+1] += mfp*r[1];
	    f_total[i*3+2] += mfp*r[2];
#if SINGLE_PASS
	    f_total[j*3+0] += -mfp*r[0];
	    f_total[j*3+1] += -mfp*r[1];
	    f_total[j*3+2] += -mfp*r[2];
#endif
	  }
	}
      }
    }
#else
    for (int j = i+1; j < numParticles; j++) {

      float *rj = &particles[j*3];
      float pj = c*(densities[j]-rho0);
      
      float r[3] = {(ri[0]-rj[0]), 
		    (ri[1]-rj[1]), 
		    (ri[2]-rj[2])};
      float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
      float mfp = -m*(pi+pj)/(densities[i]+densities[j])*gradWspiky(magr);
      // FROM FLUIDS: (does not work well)
      // float mfp = -m*(pi+pj)/(densities[i]*densities[j])*gradWspiky(magr);

      vecNormalize(r,r);

      f_total[i*3+0] += mfp*r[0];
      f_total[i*3+1] += mfp*r[1];
      f_total[i*3+2] += mfp*r[2];

      f_total[j*3+0] += -mfp*r[0];
      f_total[j*3+1] += -mfp*r[1];
      f_total[j*3+2] += -mfp*r[2];
    }
#endif
  }

  // visocity------------------------------------------------------------------
  for (int i = 0; i < numParticles; i++) {

    float *ri = &particles[i*3];
    float *vi = &velocities[i*3];
    float fviscosity[3] = {0.0f, 0.0f, 0.0f};

#if USE_GRID
    int cellCoords[3] = {int(ri[0]/cellSize), 
    			 int(ri[1]/cellSize), 
    			 int(ri[2]/cellSize)};

    int xext[2] = {cellCoords[0]-1, cellCoords[0]+1};
    if (xext[0] < 0) xext[0] = 0;
    if (xext[1] >= gridSize) xext[1] = gridSize-1;

    int yext[2] = {cellCoords[1]-1, cellCoords[1]+1};
    if (yext[0] < 0) yext[0] = 0;
    if (yext[1] >= gridSize) yext[1] = gridSize-1;

    int zext[2] = {cellCoords[2]-1, cellCoords[2]+1};
    if (zext[0] < 0) zext[0] = 0;
    if (zext[1] >= gridSize) zext[1] = gridSize-1;
	  
    for (int z = zext[0]; z <= zext[1]; z++) {
      for (int y = yext[0]; y <= yext[1]; y++) {
	for (int x = xext[0]; x <= xext[1]; x++) {
	  
	  int cellId = x + y*gridSize + z*gridSize*gridSize;

	  if (plists[cellId].empty()) {
	    continue;
	  }

	  for (int& j : plists[cellId]) {

#if SINGLE_PASS
	    if (j <= i) {
	      continue;
	    }
#endif
	    float *rj = &particles[j*3];
	    float *vj = &velocities[j*3];
	    float vdiff[3] = {vj[0]-vi[0], vj[1]-vi[1], vj[2]-vi[2]};
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);

	    float tmp = dynVisc*m*laplWviscosity(magr)*2.0f/(densities[i]+densities[j]);

	    f_total[i*3+0] += vdiff[0]*tmp;
	    f_total[i*3+1] += vdiff[1]*tmp;
	    f_total[i*3+2] += vdiff[2]*tmp;
#if SINGLE_PASS
	    f_total[j*3+0] += -vdiff[0]*tmp;
	    f_total[j*3+1] += -vdiff[1]*tmp;
	    f_total[j*3+2] += -vdiff[2]*tmp;
#endif
	  }
	}
      }
    }
#else
    for (int j = i+1; j < numParticles; j++) {

      float *rj = &particles[j*3];
      float *vj = &velocities[j*3];
      float vdiff[3] = {vj[0]-vi[0], vj[1]-vi[1], vj[2]-vi[2]};
      float r[3] = {(ri[0]-rj[0]), 
		    (ri[1]-rj[1]), 
		    (ri[2]-rj[2])};
      float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);

      float tmp = dynVisc*m*laplWviscosity(magr)*2.0f/(densities[i]+densities[j]);

      f_total[i*3+0] += vdiff[0]*tmp;
      f_total[i*3+1] += vdiff[1]*tmp;
      f_total[i*3+2] += vdiff[2]*tmp;

      f_total[j*3+0] += -vdiff[0]*tmp;
      f_total[j*3+1] += -vdiff[1]*tmp;
      f_total[j*3+2] += -vdiff[2]*tmp;
    }
#endif
  }

  // gravity-------------------------------------------------------------------
#pragma omp parallel for
  for (int i = 0; i < numParticles; i++) {

    f_total[i*3+0] += a_gravity[0]*densities[i];
    f_total[i*3+1] += a_gravity[1]*densities[i];
    f_total[i*3+2] += a_gravity[2]*densities[i];    
  }

  // surface tension-----------------------------------------------------------
  float *gradcolors = new float[numParticles*3];
  float *laplcolors = new float[numParticles];
#pragma omp parallel for
  for (int i = 0; i < numParticles; i++) {
    gradcolors[i*3+0] = 0.0f;
    gradcolors[i*3+1] = 0.0f;
    gradcolors[i*3+2] = 0.0f;
    laplcolors[i] = 0.0f;
  }

  for (int i = 0; i < numParticles; i++) {

    float *ri = &particles[i*3];
#if USE_GRID
    int cellCoords[3] = {int(ri[0]/cellSize), 
    			 int(ri[1]/cellSize), 
    			 int(ri[2]/cellSize)};

    int xext[2] = {cellCoords[0]-1, cellCoords[0]+1};
    if (xext[0] < 0) xext[0] = 0;
    if (xext[1] >= gridSize) xext[1] = gridSize-1;

    int yext[2] = {cellCoords[1]-1, cellCoords[1]+1};
    if (yext[0] < 0) yext[0] = 0;
    if (yext[1] >= gridSize) yext[1] = gridSize-1;

    int zext[2] = {cellCoords[2]-1, cellCoords[2]+1};
    if (zext[0] < 0) zext[0] = 0;
    if (zext[1] >= gridSize) zext[1] = gridSize-1;
	  
    for (int z = zext[0]; z <= zext[1]; z++) {
      for (int y = yext[0]; y <= yext[1]; y++) {
	for (int x = xext[0]; x <= xext[1]; x++) {
	  
	  int cellId = x + y*gridSize + z*gridSize*gridSize;

	  if (plists[cellId].empty()) {
	    continue;
	  }

	  for (int& j : plists[cellId]) {

#if SINGLE_PASS
	    if (j <= i) {
	      continue;
	    }
#endif
	    float *rj = &particles[j*3];
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
	    float magr = std::sqrt(r2);
	    float md = m/densities[j];
	    float maggradcolor = gradWpoly6(magr);
	    float nr[3];

	    vecNormalize(r, nr);
	    nr[0] *= maggradcolor;
	    nr[1] *= maggradcolor;
	    nr[2] *= maggradcolor;

	    gradcolors[i*3+0] += nr[0]*md;
	    gradcolors[i*3+1] += nr[1]*md;
	    gradcolors[i*3+2] += nr[2]*md;
	    laplcolors[i] += md*laplWpoly6(magr);
#if SINGLE_PASS
	    float md2 = m/densities[i];
	    gradcolors[j*3+0] += -nr[0]*md2;
	    gradcolors[j*3+1] += -nr[1]*md2;
	    gradcolors[j*3+2] += -nr[2]*md2;
	    laplcolors[j] += md*laplWpoly6(magr);
#endif
	  }
	}
      }
    }
#else
    for (int j = i+1; j < numParticles; j++) {

      float *rj = &particles[j*3];
      float r[3] = {(ri[0]-rj[0]), 
		    (ri[1]-rj[1]), 
		    (ri[2]-rj[2])};
      float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
      float magr = std::sqrt(r2);


      float md = m/densities[j];
      float md2 = m/densities[i];
      float maggradcolor = gradWpoly6(magr);
      float nr[3];
      vecNormalize(r, nr);
      nr[0] *= maggradcolor;
      nr[1] *= maggradcolor;
      nr[2] *= maggradcolor;

      gradcolors[i*3+0] += nr[0]*md;
      gradcolors[i*3+1] += nr[1]*md;
      gradcolors[i*3+2] += nr[2]*md;

      gradcolors[j*3+0] += -nr[0]*md2;
      gradcolors[j*3+1] += -nr[1]*md2;
      gradcolors[j*3+2] += -nr[2]*md2;

      laplcolors[i] += md*laplWpoly6(magr);
      laplcolors[j] += md*laplWpoly6(magr);
    }
#endif
    vecNormalize(&gradcolors[i*3], &gradcolors[i*3]);

    float tmp = -surfTension*laplcolors[i];

    f_total[i*3+0] += tmp*gradcolors[i*3+0];
    f_total[i*3+1] += tmp*gradcolors[i*3+1];
    f_total[i*3+2] += tmp*gradcolors[i*3+2];
  }

  delete [] gradcolors;
  delete [] laplcolors;

  // update velocity and advect particles--------------------------------------
#pragma omp parallel for
  for (int i = 0; i < numParticles; i++) {

    float *particle = &particles[i*3];
    float sc = timestep/densities[i];

    float *velocity = &velocities[i*3];
    
    velocity[0] += f_total[i*3+0]*sc;
    velocity[1] += f_total[i*3+1]*sc;
    velocity[2] += f_total[i*3+2]*sc;

    particle[0] = particle[0] + velocity[0]*timestep;
    particle[1] = particle[1] + velocity[1]*timestep;
    particle[2] = particle[2] + velocity[2]*timestep;
  }

  free(densities);
  free(f_total);

  // check particles on/behind the boundary------------------------------------
#pragma omp parallel for
  for (int i = 0; i < numParticles; i++) {

    float *particle = &particles[i*3];
    float *velocity = &velocities[i*3];
    float damping = 1.0f;

    if (particle[0] < 0.0f) {
      particle[0] = 0.0f;
      velocity[0] = -damping*velocity[0];
    }
    if (particle[0] >  tankSize) {
      particle[0] = tankSize;
      velocity[0] = -damping*velocity[0];
    }

    if (particle[1] < 0.0f) {
      particle[1] = 0.0f;
      velocity[1] = -damping*velocity[1];
    }
    if (particle[1] >  tankSize) {
      particle[1] = tankSize;
      velocity[1] = -damping*velocity[1];
    }

    if (particle[2] < 0.0f) {
      particle[2] = 0.0f;
      velocity[2] = -damping*velocity[2];
    }
    if (particle[2] >  tankSize) {
      particle[2] = tankSize;
      velocity[2] = -damping*velocity[2];
    }
  }
}

void createGrid()
{

}
