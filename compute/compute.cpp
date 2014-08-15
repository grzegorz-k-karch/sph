#include <time.h>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector> 
#include <forward_list>
#include <algorithm>
#include <omp.h>

#include "compute.h"

#define PARALLEL_SLABS 1
#define USE_COPIES 1

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

int cellOffsets[13];

#if USE_COPIES
typedef struct {
  float x;
  float y;
  float z;
  int idx;
} particle_t;
#endif

#if USE_COPIES
std::vector<std::forward_list<particle_t> > plists;
#else
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
  w = 45.0f/(M_PI*h6)*(h-r);
  return w;
}

float gradWspiky(const float r)
{
  float w = 0.0f;
  float hr = h - r;
  float hr2 = hr*hr;
  w = -45.0f/(M_PI*h6)*hr2;
  return w;
}

float Wpoly6(const float r)
{
  float w = 0.0f;
  float r2 = r*r;
  float hr3 = h2 - r2;
  hr3 = hr3*hr3*hr3;
  w = 315.0f/(64.0f*M_PI*h9)*hr3;
  return w;
}

float gradWpoly6(const float r)
{
  float w = 0.0f;
  float r2 = r*r;
  float hr2 = h2 - r2;
  hr2 = hr2*hr2;
  w = -945.0f/(32.0f*M_PI*h9)*r*hr2;
  return w;
}

float laplWpoly6(const float r)
{
  float w = 0.0f;
  float r2 = r*r;
  float hr = h2 - r2;
  float hr2 = hr*hr;
  w = 945.0f/(32.0f*M_PI*h9)*(4.0f*r2*hr - hr2);
  return w;
}

void initParticles(float** particles, float** velocities, int numParticles)
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

  const float cellSize = h*1.01f;
  const int gridSize = std::ceil(tankSize/cellSize);
  int nx = gridSize;
  int nxy = gridSize*gridSize;
  cellOffsets[0]  =  0;            // 0 , 0 , 0 ( 0) 
  cellOffsets[1]  =  1;            // + , 0 , 0 ( 1) 
  cellOffsets[2]  = -1 + nx;       // - , + , 0 ( 2)
  cellOffsets[3]  =    + nx;       // 0 , + , 0 ( 3)
  cellOffsets[4]  =  1 + nx;       // + , + , 0 ( 4)
  cellOffsets[5]  = -1 - nx + nxy; // - , - , + ( 5) 
  cellOffsets[6]  =    - nx + nxy; // 0 , - , + ( 6)
  cellOffsets[7]  =  1 - nx + nxy; // + , - , + ( 7)
  cellOffsets[8]  = -1      + nxy; // - , 0 , + ( 8)
  cellOffsets[9]  =         + nxy; // 0 , 0 , + ( 9)
  cellOffsets[10]  =  1      + nxy; // + , 0 , + (10)
  cellOffsets[11] = -1 + nx + nxy; // - , 0 , + (11)
  cellOffsets[12] =      nx + nxy; // 0 , + , + (12)
  cellOffsets[13] =  1 + nx + nxy; // + , + , + (13)
}

void assignParticlesToCells(float* particles, int numParticles)
{
  const float cellSize = h*1.01f;
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

#if USE_COPIES
    particle_t particle = {pos[0], pos[1], pos[2], i};
    plists[cellId].push_front(particle);
#else
    plists[cellId].push_front(i);
#endif
  }      
}

void updateParticles(float* particles, float* velocities, int numParticles)
{
  assignParticlesToCells(particles, numParticles);

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

  const float cellSize = h*1.01f;
  const int gridSize = std::ceil(tankSize/cellSize);
  const int numCells = gridSize*gridSize*gridSize;
#if PARALLEL_SLABS
  const int numCellsInSlab = gridSize*gridSize;

#pragma omp parallel for 
  for (int z = 0; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {

      int c = s + z*numCellsInSlab;
#if USE_COPIES
      for (particle_t& ri: plists[c]) {

	int i = ri.idx;
	float densityi = 0.0f;

	densities[i] += own_rho;

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (particle_t& rj : plists[cellId]) {

	    int j = rj.idx;

	    if (k == 0 && j <= i) continue;
	  
	    float r[3] = {(ri.x-rj.x), 
			  (ri.y-rj.y), 
			  (ri.z-rj.z)};
	    float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
	    float magr = std::sqrt(r2);
	    if (magr > h) continue;
	    float rho = m*Wpoly6(magr);

	    // densities[i] += rho;
	    densityi += rho;
	    densities[j] += rho;
	  }
	}
	densities[i] += densityi;
      }
#else
      for (int& i : plists[c]) {

	float densityi = 0.0f;

	float *ri = &particles[i*3];
	densities[i] += own_rho;

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (int& j : plists[cellId]) {

	    if (k == 0 && j <= i) continue;
	  
	    float *rj = &particles[j*3];
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
	    float magr = std::sqrt(r2);
	    if (magr > h) continue;
	    float rho = m*Wpoly6(magr);

	    // densities[i] += rho;
	    densityi += rho;
	    densities[j] += rho;
	  }
	}
	densities[i] += densityi;
      }
#endif
    }    
  }
#pragma omp parallel for 
  for (int z = 1; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {

      int c = s + z*numCellsInSlab;
#if USE_COPIES
      for (particle_t& ri: plists[c]) {

	int i = ri.idx;
	float densityi = 0.0f;

	densities[i] += own_rho;

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (particle_t& rj : plists[cellId]) {

	    int j = rj.idx;

	    if (k == 0 && j <= i) continue;
	  
	    float r[3] = {(ri.x-rj.x), 
			  (ri.y-rj.y), 
			  (ri.z-rj.z)};
	    float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
	    float magr = std::sqrt(r2);
	    if (magr > h) continue;
	    float rho = m*Wpoly6(magr);

	    // densities[i] += rho;
	    densityi += rho;
	    densities[j] += rho;
	  }
	}
	densities[i] += densityi;
      }
#else
      for (int& i : plists[c]) {

	float densityi = 0.0f;

	float *ri = &particles[i*3];
	densities[i] += own_rho;

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (int& j : plists[cellId]) {

	    if (k == 0 && j <= i) continue;
	  
	    float *rj = &particles[j*3];
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
	    float magr = std::sqrt(r2);
	    if (magr > h) continue;
	    float rho = m*Wpoly6(magr);

	    // densities[i] += rho;
	    densityi += rho;
	    densities[j] += rho;
	  }
	}
	densities[i] += densityi;
      }
#endif
    }    
  }
#else
  for (int c = 0; c < numCells; c++) {
    for (int& i : plists[c]) {

      float *ri = &particles[i*3];
      densities[i] += own_rho;

      for (int k = 0; k < 14; k++) {

  	int cellId = c + cellOffsets[k];
  	if (cellId >= numCells) continue;
	
  	for (int& j : plists[cellId]) {

  	  if (k == 0 && j <= i) continue;
	  
  	  float *rj = &particles[j*3];
  	  float r[3] = {(ri[0]-rj[0]), 
  			(ri[1]-rj[1]), 
  			(ri[2]-rj[2])};
  	  float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
  	  float magr = std::sqrt(r2);
  	  if (magr > h) continue;
  	  float rho = m*Wpoly6(magr);

  	  densities[i] += rho;
  	  densities[j] += rho;
  	}
      }
    }
  }
#endif
  static int testStep = 1;
  static int done = 0;

  if (done == testStep) {
    std::cout << "densities[0] = " << densities[0] << std::endl;
  }
  done++;

  // pressure-----------------------------------------------------------------
  float cs = soundSpeed*soundSpeed;

#if PARALLEL_SLABS
#pragma omp parallel for 
  for (int z = 0; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {      

      int c = s + z*numCellsInSlab;
#if USE_COPIES
      for (particle_t& ri : plists[c]) {
	
	int i = ri.idx;
	float pi = cs*(densities[i]-rho0);

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (particle_t& rj : plists[cellId]) {

	    int j = rj.idx;
	    if (k == 0 && j <= i) continue;

	    float pj = cs*(densities[j]-rho0);
      	    float r[3] = {(ri.x-rj.x), 
			  (ri.y-rj.y), 
			  (ri.z-rj.z)};
	    float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
	    if (magr > h) continue;
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
	}
      }
#else
      for (int& i : plists[c]) {
	
	float *ri = &particles[i*3];
	float pi = cs*(densities[i]-rho0);

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (int& j : plists[cellId]) {

	    if (k == 0 && j <= i) continue;

	    float *rj = &particles[j*3];
	    float pj = cs*(densities[j]-rho0);
      
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
	    if (magr > h) continue;
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
	}
      }
#endif
    }
  }
#pragma omp parallel for 
  for (int z = 1; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {      

      int c = s + z*numCellsInSlab;
#if USE_COPIES
      for (particle_t& ri : plists[c]) {
	
	int i = ri.idx;
	float pi = cs*(densities[i]-rho0);

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (particle_t& rj : plists[cellId]) {

	    int j = rj.idx;
	    if (k == 0 && j <= i) continue;

	    float pj = cs*(densities[j]-rho0);
      	    float r[3] = {(ri.x-rj.x), 
			  (ri.y-rj.y), 
			  (ri.z-rj.z)};
	    float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
	    if (magr > h) continue;
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
	}
      }
#else
      for (int& i : plists[c]) {
	
	float *ri = &particles[i*3];
	float pi = cs*(densities[i]-rho0);

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (int& j : plists[cellId]) {

	    if (k == 0 && j <= i) continue;

	    float *rj = &particles[j*3];
	    float pj = cs*(densities[j]-rho0);
      
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
	    if (magr > h) continue;
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
	}
      }
#endif
    }
  }
#else
  for (int c = 0; c < numCells; c++) {
    for (int& i : plists[c]) {
	
      float *ri = &particles[i*3];
      float pi = cs*(densities[i]-rho0);

      for (int k = 0; k < 14; k++) {

	int cellId = c + cellOffsets[k];
	if (cellId >= numCells) continue;
	
	for (int& j : plists[cellId]) {

	  if (k == 0 && j <= i) continue;

	  float *rj = &particles[j*3];
	  float pj = cs*(densities[j]-rho0);
      
	  float r[3] = {(ri[0]-rj[0]), 
			(ri[1]-rj[1]), 
			(ri[2]-rj[2])};
	  float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
	  if (magr > h) continue;
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
      }
    }
  }
#endif
  // visocity------------------------------------------------------------------
#if PARALLEL_SLABS
#pragma omp parallel for 
  for (int z = 0; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {      

      int c = s + z*numCellsInSlab;
#if USE_COPIES
      for (particle_t& ri : plists[c]) {

	float *ri = &particles[i*3];
	float *vi = &velocities[i*3];
	float fviscosity[3] = {0.0f, 0.0f, 0.0f};

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (int& j : plists[cellId]) {

	    if (k == 0 && j <= i) continue;

	    float *rj = &particles[j*3];
	    float *vj = &velocities[j*3];
	    float vdiff[3] = {vj[0]-vi[0], vj[1]-vi[1], vj[2]-vi[2]};
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
	    if (magr > h) continue;
	    float tmp = dynVisc*m*laplWviscosity(magr)*2.0f/(densities[i]+densities[j]);

	    f_total[i*3+0] += vdiff[0]*tmp;
	    f_total[i*3+1] += vdiff[1]*tmp;
	    f_total[i*3+2] += vdiff[2]*tmp;

	    f_total[j*3+0] += -vdiff[0]*tmp;
	    f_total[j*3+1] += -vdiff[1]*tmp;
	    f_total[j*3+2] += -vdiff[2]*tmp;
	  }
	}
      }
#else
      for (int& i : plists[c]) {
	
	float *ri = &particles[i*3];
	float *vi = &velocities[i*3];
	float fviscosity[3] = {0.0f, 0.0f, 0.0f};

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (int& j : plists[cellId]) {

	    if (k == 0 && j <= i) continue;

	    float *rj = &particles[j*3];
	    float *vj = &velocities[j*3];
	    float vdiff[3] = {vj[0]-vi[0], vj[1]-vi[1], vj[2]-vi[2]};
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
	    if (magr > h) continue;
	    float tmp = dynVisc*m*laplWviscosity(magr)*2.0f/(densities[i]+densities[j]);

	    f_total[i*3+0] += vdiff[0]*tmp;
	    f_total[i*3+1] += vdiff[1]*tmp;
	    f_total[i*3+2] += vdiff[2]*tmp;

	    f_total[j*3+0] += -vdiff[0]*tmp;
	    f_total[j*3+1] += -vdiff[1]*tmp;
	    f_total[j*3+2] += -vdiff[2]*tmp;
	  }
	}
      }
#endif
    }
  }
#pragma omp parallel for
  for (int z = 1; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {      
      int c = s + z*numCellsInSlab;

      for (int& i : plists[c]) {
	
	float *ri = &particles[i*3];
	float *vi = &velocities[i*3];
	float fviscosity[3] = {0.0f, 0.0f, 0.0f};

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (int& j : plists[cellId]) {

	    if (k == 0 && j <= i) continue;

	    float *rj = &particles[j*3];
	    float *vj = &velocities[j*3];
	    float vdiff[3] = {vj[0]-vi[0], vj[1]-vi[1], vj[2]-vi[2]};
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
	    if (magr > h) continue;
	    float tmp = dynVisc*m*laplWviscosity(magr)*2.0f/(densities[i]+densities[j]);

	    f_total[i*3+0] += vdiff[0]*tmp;
	    f_total[i*3+1] += vdiff[1]*tmp;
	    f_total[i*3+2] += vdiff[2]*tmp;

	    f_total[j*3+0] += -vdiff[0]*tmp;
	    f_total[j*3+1] += -vdiff[1]*tmp;
	    f_total[j*3+2] += -vdiff[2]*tmp;
	  }
	}
      }
    }
  }
#else
  for (int c = 0; c < numCells; c++) {
    for (int& i : plists[c]) {
	
      float *ri = &particles[i*3];
      float *vi = &velocities[i*3];
      float fviscosity[3] = {0.0f, 0.0f, 0.0f};

      for (int k = 0; k < 14; k++) {

	int cellId = c + cellOffsets[k];
	if (cellId >= numCells) continue;
	
	for (int& j : plists[cellId]) {

	  if (k == 0 && j <= i) continue;

	  float *rj = &particles[j*3];
	  float *vj = &velocities[j*3];
	  float vdiff[3] = {vj[0]-vi[0], vj[1]-vi[1], vj[2]-vi[2]};
	  float r[3] = {(ri[0]-rj[0]), 
			(ri[1]-rj[1]), 
			(ri[2]-rj[2])};
	  float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
	  if (magr > h) continue;
	  float tmp = dynVisc*m*laplWviscosity(magr)*2.0f/(densities[i]+densities[j]);

	  f_total[i*3+0] += vdiff[0]*tmp;
	  f_total[i*3+1] += vdiff[1]*tmp;
	  f_total[i*3+2] += vdiff[2]*tmp;

	  f_total[j*3+0] += -vdiff[0]*tmp;
	  f_total[j*3+1] += -vdiff[1]*tmp;
	  f_total[j*3+2] += -vdiff[2]*tmp;
	}
      }
    }
  }
#endif
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
#if PARALLEL_SLABS
#pragma omp parallel for 
  for (int z = 0; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {      
      int c = s + z*numCellsInSlab;

      for (int& i : plists[c]) {

	float *ri = &particles[i*3];

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (int& j : plists[cellId]) {

	    if (k == 0 && j <= i) continue;

	    float *rj = &particles[j*3];
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
	    float magr = std::sqrt(r2);
	    if (magr > h) continue;
	    float md = m*2.0f/(densities[i]+densities[j]);
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

	    gradcolors[j*3+0] += -nr[0]*md;
	    gradcolors[j*3+1] += -nr[1]*md;
	    gradcolors[j*3+2] += -nr[2]*md;
	    laplcolors[j] += md*laplWpoly6(magr);
	  }
	}	
      }
    }
  }
#pragma omp parallel for 
  for (int z = 1; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {      
      int c = s + z*numCellsInSlab;
      
      for (int& i : plists[c]) {
  
	float *ri = &particles[i*3];
  
	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (int& j : plists[cellId]) {

	    if (k == 0 && j <= i) continue;

	    float *rj = &particles[j*3];
	    float r[3] = {(ri[0]-rj[0]), 
			  (ri[1]-rj[1]), 
			  (ri[2]-rj[2])};
	    float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
	    float magr = std::sqrt(r2);
	    if (magr > h) continue;
	    float md = m*2.0f/(densities[i]+densities[j]);
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

	    gradcolors[j*3+0] += -nr[0]*md;
	    gradcolors[j*3+1] += -nr[1]*md;
	    gradcolors[j*3+2] += -nr[2]*md;
	    laplcolors[j] += md*laplWpoly6(magr);
	  }
	}	
      }
    }
  }
#else
  for (int c = 0; c < numCells; c++) {
    for (int& i : plists[c]) {

      float *ri = &particles[i*3];

      for (int k = 0; k < 14; k++) {

	int cellId = c + cellOffsets[k];
	if (cellId >= numCells) continue;
	
	for (int& j : plists[cellId]) {

	  if (k == 0 && j <= i) continue;

	  float *rj = &particles[j*3];
	  float r[3] = {(ri[0]-rj[0]), 
			(ri[1]-rj[1]), 
			(ri[2]-rj[2])};
	  float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
	  float magr = std::sqrt(r2);
	  if (magr > h) continue;
	  float md = m*2.0f/(densities[i]+densities[j]);
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

	  gradcolors[j*3+0] += -nr[0]*md;
	  gradcolors[j*3+1] += -nr[1]*md;
	  gradcolors[j*3+2] += -nr[2]*md;
	  laplcolors[j] += md*laplWpoly6(magr);
	}
      }	
    }
  }
#endif

#pragma omp parallel for
  for (int i = 0; i < numParticles; i++) {

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
