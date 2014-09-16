#include <time.h>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector> 
#include <forward_list>
#include <algorithm>
#include <omp.h>
#include <helper_math.h>

#include "compute.h"

using std::size_t;
using std::vector;
using std::cout;
using std::endl;

#define PCISPH 1

float soundSpeed = 1.0f;
float m = 0.00020543f;
float rho0 = 600.0f;
float dynVisc = 0.2f;
float a_gravity[3] = {0.0f, -10.0f, 0.0f};
float timestep = 0.0025f;
float tankSize = 0.4154f;//0.2154f;
float surfTension = 0.0189f;//378f;//0.0728f;

float h = 0.01f;
float h9 = 0.0f;
float h2 = 0.0f;
float h6 = 0.0f;

const float rhoError = 0.01f;
const int minNumIter = 3;
const int maxNumIter = 10;

int cellOffsets[14];

typedef struct {

  // position
  float x;
  float y;
  float z;

  // index
  int idx;

  // velocity
  float u;
  float v;
  float w;  

  // density
  float rho;

  // force
  float fx;
  float fy;
  float fz;
  float dummy;

  // surface tension
  float gx;
  float gy;
  float gz;
  float lc;

} particle_t;
#if PCISPH
std::vector<float> pressures;
#endif
std::vector<std::vector<particle_t> > plists;

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

int checkDensityError()
{
  for (const auto& cell : plists) {
    for (const auto& particle : cell) {

      float err = std::abs(particle.rho-rho0);
      if (err > rhoError) {
	return 1;
      }
    }
  }
  return 0;
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

#if PCISPH
  pressures.resize(numParticles);
#endif
}

void deleteParticles(float** particles, float** velocities)
{
  delete [] *particles;
  delete [] *velocities;
}

void assignParticlesToCells(float* particles, float* velocities, 
			    int numParticles)
{
  const float cellSize = h*1.01f;
  const int gridSize = std::ceil(tankSize/cellSize);
  const int numCells = gridSize*gridSize*gridSize;

  for (auto& cell : plists) {
    cell.clear();
  }
  // for (int i = 0; i < plists.size(); i++) {
  //   plists[i].clear();
  // }
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

    float *velo = &velocities[i*3];

    particle_t particle = {pos[0], pos[1], pos[2], i, 
			   velo[0], velo[1], velo[2], 0.0f,
			   0.0f, 0.0f, 0.0f, 0.0f,
			   0.0f, 0.0f, 0.0f, 0.0f};
    plists[cellId].push_back(particle);
  }      
}

void computeDensity(int z0)
{
  // compute density for each particle-----------------------------------------
  float own_rho = m*Wpoly6(0.0f);

  const float cellSize = h*1.01f;
  const int gridSize = std::ceil(tankSize/cellSize);
  const int numCells = gridSize*gridSize*gridSize;
  const int numCellsInSlab = gridSize*gridSize;

#pragma omp parallel for 
  for (int z = z0; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {

      int c = s + z*numCellsInSlab;

      for (particle_t& ri: plists[c]) {

	int i = ri.idx;
	float densityi = 0.0f;

	ri.rho += own_rho;

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (particle_t& rj : plists[cellId]) {

	    int j = rj.idx;

	    if (k == 0 && j >= i) continue;
	  
	    float r[3] = {(ri.x-rj.x), 
			  (ri.y-rj.y), 
			  (ri.z-rj.z)};
	    float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
	    float magr = std::sqrt(r2);
	    if (magr > h) continue;
	    float rho = m*Wpoly6(magr);

	    densityi += rho;
	    rj.rho += rho;
	  }
	}
	ri.rho += densityi;
      }
    }    
  }
}

void computeDensityComplete()
{
  computeDensity(0);
  computeDensity(1);
}

void computePressureForces(int z0)
{
  const float cellSize = h*1.01f;
  const int gridSize = std::ceil(tankSize/cellSize);
  const int numCells = gridSize*gridSize*gridSize;
  const int numCellsInSlab = gridSize*gridSize;

  float cs = soundSpeed*soundSpeed;

#pragma omp parallel for 
  for (int z = z0; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {      

      int c = s + z*numCellsInSlab;

      for (particle_t& ri : plists[c]) {
	
	int i = ri.idx;
	float pi = cs*(ri.rho-rho0);

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (particle_t& rj : plists[cellId]) {

	    int j = rj.idx;

	    if (k == 0 && j >= i) continue;

	    float pj = cs*(rj.rho-rho0);
      	    float r[3] = {(ri.x-rj.x), (ri.y-rj.y), (ri.z-rj.z)};
	    float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
	    if (magr > h) continue;

	    // pressure
	    float mfp = -m*(pi+pj)/(ri.rho+rj.rho)*gradWspiky(magr);
	    // FROM FLUIDS: (does not work well)
	    // float mfp = -m*(pi+pj)/(densities[i]*densities[j])*gradWspiky(magr);

	    vecNormalize(r,r);
	    
	    ri.fx += mfp*r[0];
	    ri.fy += mfp*r[1];
	    ri.fz += mfp*r[2];

	    rj.fx += -mfp*r[0];
	    rj.fy += -mfp*r[1];
	    rj.fz += -mfp*r[2];
	  }
	}
      }
    }
  }
}

void computePressureForcesComplete()
{
  computePressureForces(0);
  computePressureForces(1);
}

void computeOtherForces(int z0)
{
  const float cellSize = h*1.01f;
  const int gridSize = std::ceil(tankSize/cellSize);
  const int numCells = gridSize*gridSize*gridSize;
  const int numCellsInSlab = gridSize*gridSize;

#pragma omp parallel for 
  for (int z = z0; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {      

      int c = s + z*numCellsInSlab;

      for (particle_t& ri : plists[c]) {
	
	int i = ri.idx;

	for (int k = 0; k < 14; k++) {

	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	
	  for (particle_t& rj : plists[cellId]) {

	    int j = rj.idx;

	    if (k == 0 && j >= i) continue;

      	    float r[3] = {(ri.x-rj.x), (ri.y-rj.y), (ri.z-rj.z)};
	    float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
	    if (magr > h) continue;

	    vecNormalize(r,r);

	    // viscosity
	    float tmp = dynVisc*m*laplWviscosity(magr)*2.0f/(ri.rho+rj.rho);
	    float vdiff[3] = {rj.u - ri.u, rj.v - ri.v, rj.w - ri.w};

	    ri.fx += vdiff[0]*tmp;
	    ri.fy += vdiff[1]*tmp;
	    ri.fz += vdiff[2]*tmp;

	    rj.fx += -vdiff[0]*tmp;
	    rj.fy += -vdiff[1]*tmp;
	    rj.fz += -vdiff[2]*tmp;

	    // surface tension
	    float md = m*2.0f/(ri.rho+rj.rho);
	    float maggradcolor = gradWpoly6(magr);

	    r[0] *= maggradcolor;
	    r[1] *= maggradcolor;
	    r[2] *= maggradcolor;

	    ri.gx += r[0]*md;
	    ri.gy += r[1]*md;
	    ri.gz += r[2]*md;
	    ri.lc += md*laplWpoly6(magr);

	    rj.gx += -r[0]*md;
	    rj.gy += -r[1]*md;
	    rj.gz += -r[2]*md;
	    rj.lc += md*laplWpoly6(magr);
	  }
	}
      }
    }
  }
}

void finalizeOtherForces()
{
#pragma omp parallel for
  for (int i = 0; i < plists.size(); i++) {
    for (int j = 0; j < plists[i].size(); j++) {

      // gravity
      plists[i][j].fx += a_gravity[0]*plists[i][j].rho;
      plists[i][j].fy += a_gravity[1]*plists[i][j].rho;
      plists[i][j].fz += a_gravity[2]*plists[i][j].rho;

      // surface tension
      float gradcolor[3] = {plists[i][j].gx,
			    plists[i][j].gy,
			    plists[i][j].gz};
      vecNormalize(gradcolor, gradcolor);

      float tmp = -surfTension*plists[i][j].lc;

      plists[i][j].fx += tmp*gradcolor[0];
      plists[i][j].fy += tmp*gradcolor[1];
      plists[i][j].fz += tmp*gradcolor[2];  
    }
  }
}

void computeOtherForcesComplete()
{
  computeOtherForces(0);
  computeOtherForces(1);
  finalizeOtherForces();
}

void integrate(float* particles, float* velocities)
{
#pragma omp parallel for
  for (int i = 0; i < plists.size(); i++) {
    for (int j = 0; j < plists[i].size(); j++) {

      int idx = plists[i][j].idx;

      float *particle = &particles[idx*3];
      float sc = timestep/plists[i][j].rho;

      float *velocity = &velocities[idx*3];
    
      velocity[0] += plists[i][j].fx*sc;
      velocity[1] += plists[i][j].fy*sc;
      velocity[2] += plists[i][j].fz*sc;

      particle[0] = particle[0] + velocity[0]*timestep;
      particle[1] = particle[1] + velocity[1]*timestep;
      particle[2] = particle[2] + velocity[2]*timestep;      
    }
  }
}

void checkBoundaries(float* particles, float* velocities, int numParticles)
{
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

#if PCISPH
vector<float3> prod_sum_grad;
vector<float> sum_prod_grad;
void predictPressure(int z0)
{
  const float cellSize = h*1.01f;
  const int gridSize = std::ceil(tankSize/cellSize);
  const int numCells = gridSize*gridSize*gridSize;
  const int numCellsInSlab = gridSize*gridSize;

#pragma omp parallel for 
  for (int z = z0; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {

      int c = s + z*numCellsInSlab;

      for (particle_t& ri: plists[c]) {

	int i = ri.idx;

	for (int k = 0; k < 14; k++) {
	  
	  int cellId = c + cellOffsets[k];
	  if (cellId >= numCells) continue;
	  
	  for (particle_t& rj : plists[cellId]) {

	    int j = rj.idx;
	    if (k == 0 && j >= i) continue;
	  
	    float r[3] = {(ri.x-rj.x), (ri.y-rj.y), (ri.z-rj.z)};
	    float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
  	    float magr = std::sqrt(r2);
	    if (magr > h) continue;

	    float gw = gradWspiky(magr);
	    float gw2 = gw*gw;

	    sum_prod_grad[i] += gw2;
	    sum_prod_grad[j] += gw2;

	    vecNormalize(r,r);
	    float3 gwv = make_float3(r[0]*gw, r[1]*gw, r[2]*gw);
	    prod_sum_grad[i] += gwv;
	    prod_sum_grad[j] -= gwv;
	  }
	}
      }
    }
  }
}

void predictPressureComplete()
{
  sum_prod_grad = vector<float>(pressures.size(), 0.0f);
  prod_sum_grad = vector<float3>(pressures.size(), make_float3(0.0f));

  predictPressure(0);
  predictPressure(1);

  const float beta = timestep*timestep*m*2.0f/(rho0+rho0);
  vector<float> delta(pressures.size(), 0.0f);

  //#pragma omp parallel for 
  for (int i = 0; i < pressures.size(); i++) {

    float divisor = beta*(-dot(prod_sum_grad[i], prod_sum_grad[i]) - sum_prod_grad[i]);
    if (divisor != 0.0f) 
      //      cout << beta << " " << dot(prod_sum_grad[i], prod_sum_grad[i]) << " " << sum_prod_grad[i] << endl;
      delta[i] = -1.0f/divisor;
  }
  //#pragma omp parallel for 
  for (int i = 0; i < plists.size(); i++) {
    for (int j = 0; j < plists[i].size(); j++) {
      
      int idx = plists[i][j].idx;
      pressures[idx] += (plists[i][j].rho - rho0)*delta[idx];
    }
  }
}

void predictPressureForces()
{
  for (int i = 0; i < plists.size(); i++) {
    for (int j = 0; j < plists[i].size(); j++) {

      int idx = plists[i][j].idx;
      float rho02 = rho0+rho0;
      float tmp = -m*2.0f*pressures[idx]/rho02;
      plists[i][j].fx += tmp*prod_sum_grad[idx].x;
      plists[i][j].fy += tmp*prod_sum_grad[idx].y;
      plists[i][j].fz += tmp*prod_sum_grad[idx].z;
    }
  }
}

void initPressures()
{
  for (float& p : pressures) {
    p = 0.0f;
  }
}
#endif

#if PCISPH
void updateParticles(float* particles, float* velocities, int numParticles)
{
  assignParticlesToCells(particles, velocities, numParticles);
  
  computeDensityComplete();
  computeOtherForcesComplete();
  initPressures();

  int iter = 0;
  while ((checkDensityError() || iter < minNumIter) && iter < maxNumIter) {

    integrate(particles, velocities);
    computeDensityComplete();
    predictPressureComplete();
    predictPressureForces();

    iter++;
  }

  // computePressureForces(0);
  // computePressureForces(1);  

  // integrate(particles, velocities);
  checkBoundaries(particles, velocities, numParticles);
}
#else
void updateParticles(float* particles, float* velocities, int numParticles)
{
  assignParticlesToCells(particles, velocities, numParticles);

  computeDensityComplete();
  computePressureForcesComplete();
  computeOtherForcesComplete();
  integrate(particles, velocities);
  checkBoundaries(particles, velocities, numParticles);
}
#endif

