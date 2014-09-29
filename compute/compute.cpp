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


#define REMOVESTRUCT 0
#define PCISPH 0

#if PCISPH
float delta;
#endif

using std::size_t;
using std::vector;
using std::cout;
using std::endl;

float soundSpeed = 1.0f;
float m = 0.00020543f;
float rho0 = 600.0f;
float dynVisc = 0.2f;
float3 a_gravity;
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

#if REMOVESTRUCT
vector<vector<float4>>    X; // Position
vector<vector<float4>>    U; // Velocity
vector<vector<unsigned>> Id; // Index
vector<vector<float>>   Rho; // Denity
vector<vector<float4>>    F; // Force
vector<vector<float4>>    K; // surface tension
#else
typedef struct {

  // position
  float x;
  float y;
  float z;

  // index
  unsigned idx;

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

std::vector<std::vector<particle_t> > plists;
#endif//REMOVESTRUCT
// kernel coefficients
float laplWviscosityCoeff;
float gradWspikyCoeff;
float Wpoly6Coeff;
float gradWpoly6Coeff;
float laplWpoly6Coeff;

float laplWviscosity(const float r)
{
  float w = 0.0f;
  w = laplWviscosityCoeff*(h-r);
  return w;
}

float gradWspiky(const float r)
{
  float w = 0.0f;
  float hr = h - r;
  float hr2 = hr*hr;
  w = gradWspikyCoeff*hr2;
  return w;
}

float Wpoly6(const float r)
{
  float w = 0.0f;
  float r2 = r*r;
  float hr3 = h2 - r2;
  hr3 = hr3*hr3*hr3;
  w = Wpoly6Coeff*hr3;
  return w;
}

float gradWpoly6(const float r)
{
  float w = 0.0f;
  float r2 = r*r;
  float hr2 = h2 - r2;
  hr2 = hr2*hr2;
  w = gradWpoly6Coeff*r*hr2;
  return w;
}

float laplWpoly6(const float r)
{
  float w = 0.0f;
  float r2 = r*r;
  float hr = h2 - r2;
  float hr2 = hr*hr;
  w = laplWpoly6Coeff*(4.0f*r2*hr - hr2);
  return w;
}

#if PCISPH
void precomputeDelta()
{
  const unsigned nn = 32;
  float x[] = { 1, 1, 0,-1,-1,-1, 0, 1, 1, 1, 0,-1,-1,-1, 0, 1, 
		  1, 1, 0,-1,-1,-1, 0, 1, 2,-2, 0, 0, 0, 0, 0, 0 };
  float y[] = { 0, 1, 1, 1, 0,-1,-1,-1, 0, 1, 1, 1, 0,-1,-1,-1, 
		  0, 1, 1, 1, 0,-1,-1,-1, 0, 0, 2,-2, 0, 0, 0, 0 };
  float z[] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
	         -1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 2,-2, 1,-1 };

  const float particleRadius = std::pow(m/rho0,1.0/3.0)*0.75f;
  // const float a = m*timestep/rho0;
  // const float beta = 2.0f*a*a;
  // beta derived from force computation from the original sph paper
  const float beta = timestep*timestep*m/(2.0f*rho0);

  float3 sum_grd = make_float3(0.0f);
  float sum_dot_grd = 0.0f;

  for (int i = 0; i < nn; i++) {

    float3 x_j = make_float3(x[i]*particleRadius,y[i]*particleRadius,z[i]*particleRadius);
    float dist = length(x_j);
    float grd = gradWspiky(dist);
    float3 r = normalize(x_j);
    
    sum_grd += grd*r;
    sum_dot_grd += dot(grd*r,grd*r);
  }

  delta = 1.0f/(beta*(-dot(sum_grd,sum_grd) - sum_dot_grd));
}
#endif

void initParticles(std::vector<float4>& particles, 
		   std::vector<float4>& velocities, 
		   unsigned numParticles)
{
  h2 = h*h;
  h6 = h2*h2*h2;
  h9 = h*h2*h6;

  a_gravity = make_float3(0.0f, -10.0f, 0.0f);

  laplWviscosityCoeff = 45.0f/(M_PI*h6);
  gradWspikyCoeff = -45.0f/(M_PI*h6);
  Wpoly6Coeff = 315.0f/(64.0f*M_PI*h9);
  gradWpoly6Coeff = -945.0f/(32.0f*M_PI*h9);
  laplWpoly6Coeff = 945.0f/(32.0f*M_PI*h9);

  particles.resize(numParticles);
  velocities.resize(numParticles);

  srand(0);

  for (unsigned i = 0; i < numParticles; i++) {

    particles[i].x = float(rand())/RAND_MAX*tankSize/2.0f;
    particles[i].y = float(rand())/RAND_MAX*tankSize/2.0f;
    particles[i].z = float(rand())/RAND_MAX*tankSize/2.0f;
    particles[i].w = 1.0f;

    velocities[i] = make_float4(0.0f);
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
  cellOffsets[10]  =  1     + nxy; // + , 0 , + (10)
  cellOffsets[11] = -1 + nx + nxy; // - , 0 , + (11)
  cellOffsets[12] =      nx + nxy; // 0 , + , + (12)
  cellOffsets[13] =  1 + nx + nxy; // + , + , + (13)

#if PCISPH
  precomputeDelta();
#endif
}

#if REMOVESTRUCT
void assignParticlesToCells(std::vector<float4>& particles, 
			    std::vector<float4>& velocities)
{
  const float cellSize = h*1.01f;
  const int gridSize = std::ceil(tankSize/cellSize);
  const int numCells = gridSize*gridSize*gridSize;

  X.resize(numCells);
  U.resize(numCells);
  Id.resize(numCells);
  Rho.resize(numCells);
  F.resize(numCells);
  K.resize(numCells);

  for (unsigned i = 0; i < numCells; i++) {
    X[i].clear();
    U[i].clear();
    Id[i].clear();
    Rho[i].clear();
    F[i].clear();
    K[i].clear();
  }

  const float4 zerof4 = make_float4(0.0);
  const unsigned numParticles = particles.size();

  for (unsigned i = 0; i < numParticles; i++) {

    float4 pos = particles[i];
    int cellCoords[3] = {int(pos.x/cellSize), 
			 int(pos.y/cellSize), 
			 int(pos.z/cellSize)};
    int cellId = cellCoords[0] + 
      cellCoords[1]*gridSize + 
      cellCoords[2]*gridSize*gridSize;

    X[cellId].push_back(pos);
    U[cellId].push_back(velocities[i]);
    Id[cellId].push_back(i);
    Rho[cellId].push_back(0.0f);
    F[cellId].push_back(zerof4);
    K[cellId].push_back(zerof4);
  }
}
#else
void assignParticlesToCells(std::vector<float4>& particles, 
			    std::vector<float4>& velocities)
{
  const float cellSize = h*1.01f;
  const int gridSize = std::ceil(tankSize/cellSize);
  const int numCells = gridSize*gridSize*gridSize;

  for (auto& cell : plists) {
    cell.clear();
  }
  plists.clear();
  plists.resize(numCells);

  unsigned numParticles = particles.size();
  for (unsigned i = 0; i < numParticles; i++) {

    float4 pos = particles[i];
    int cellCoords[3] = {int(pos.x/cellSize), 
			 int(pos.y/cellSize), 
			 int(pos.z/cellSize)};
    int cellId = 
      cellCoords[0] + 
      cellCoords[1]*gridSize + 
      cellCoords[2]*gridSize*gridSize;

    float4 velo = velocities[i];

    particle_t particle = {pos.x, pos.y, pos.z, i, 
			   velo.x, velo.y, velo.z, 0.0f,
			   0.0f, 0.0f, 0.0f, 0.0f,
			   0.0f, 0.0f, 0.0f, 0.0f};
    plists[cellId].push_back(particle);
  }      
}
#endif//REMOVESTRUCT

#if REMOVESTRUCT
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

      int cell = s + z*numCellsInSlab;
      unsigned numParticlesInCell = X[cell].size();
      for (unsigned i = 0; i < numParticlesInCell; i++) {

	float densityi = 0.0f;

	Rho[cell][i] += own_rho;

	for (int k = 0; k < 14; k++) {

	  int neighborCell = cell + cellOffsets[k];
	  if (neighborCell >= numCells) {
	    continue;
	  }

	  unsigned numParticlesInNeighborCell = X[neighborCell].size();
	  for (unsigned j = 0; j < numParticlesInNeighborCell; j++) {

	    if (k == 0 && Id[neighborCell][j] >= Id[cell][i]) {
	      continue;
	    }
	    float4 r = X[cell][i] - X[neighborCell][j];
	    float r2 = r.x*r.x+r.y*r.y+r.z*r.z;
	    float magr = std::sqrt(r2);
	    if (magr > h) {
	      continue;
	    }
	    float rho = m*Wpoly6(magr);

	    densityi += rho;
	    Rho[neighborCell][j] += rho;
	  }
	}
	Rho[cell][i] += densityi;
      }
    }    
  }
}
#else
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
#endif//REMOVESTRUCT
void computeDensityComplete()
{
  computeDensity(0);
  computeDensity(1);
}
#if REMOVESTRUCT
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

      int cell = s + z*numCellsInSlab;
      unsigned numParticlesInCell = X[cell].size();
      for (unsigned i = 0; i < numParticlesInCell; i++) {
	
	float rho_i = Rho[cell][i];
	float pi = cs*(rho_i-rho0);

	for (int k = 0; k < 14; k++) {

	  int neighborCell = cell + cellOffsets[k];
	  if (neighborCell >= numCells) {
	    continue;
	  }

	  unsigned numParticlesInNeighborCell = X[neighborCell].size();
	  for (unsigned j = 0; j < numParticlesInNeighborCell; j++) {

	    if (k == 0 && Id[neighborCell][j] >= Id[cell][i]) {
	      continue;
	    }

	    float rho_j = Rho[neighborCell][j];
	    float pj = cs*(rho_j-rho0);
      	    float3 r = make_float3(X[cell][i] - X[neighborCell][j]);
	    float magr = length(r);
	    if (magr > h) {
	      continue;
	    }
	    // pressure
	    float mfp = -m*(pi+pj)/(rho_i+rho_j)*gradWspiky(magr);
	    // FROM FLUIDS: (does not work well)
	    // float mfp = -m*(pi+pj)/(densities[i]*densities[j])*gradWspiky(magr);
	    // float mfp = -m*m*(pi/(ri.rho*ri.rho)+pj/(rj.rho*rj.rho))*gradWspiky(magr);

	    if (magr > 0.0f) {
	      r = normalize(r);
	    }
	    F[cell][i] += make_float4(mfp*r);
	    F[neighborCell][j] += make_float4(-mfp*r);
	  }
	}
      }
    }
  }
}
#else
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
      	    float3 r = make_float3(ri.x-rj.x, ri.y-rj.y, ri.z-rj.z);
	    float magr = length(r);
	    if (magr > h) continue;

	    // pressure
	    float mfp = -m*(pi+pj)/(ri.rho+rj.rho)*gradWspiky(magr);
	    // FROM FLUIDS: (does not work well)
	    // float mfp = -m*(pi+pj)/(densities[i]*densities[j])*gradWspiky(magr);
	    // float mfp = -m*m*(pi/(ri.rho*ri.rho)+pj/(rj.rho*rj.rho))*gradWspiky(magr);

	    if (magr > 0.0f) {
	      r = normalize(r);
	    }
	    ri.fx += mfp*r.x;
	    ri.fy += mfp*r.y;
	    ri.fz += mfp*r.z;

	    rj.fx += -mfp*r.x;
	    rj.fy += -mfp*r.y;
	    rj.fz += -mfp*r.z;
	  }
	}
      }
    }
  }
}
#endif//REMOVESTRUCT
void computePressureForcesComplete()
{
  computePressureForces(0);
  computePressureForces(1);
}
#if REMOVESTRUCT
void computeOtherForces(int z0)
{
  const float cellSize = h*1.01f;
  const int gridSize = std::ceil(tankSize/cellSize);
  const int numCells = gridSize*gridSize*gridSize;
  const int numCellsInSlab = gridSize*gridSize;

#pragma omp parallel for 
  for (int z = z0; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {      

      int cell = s + z*numCellsInSlab;
      unsigned numParticlesInCell = X[cell].size();
      for (unsigned i = 0; i < numParticlesInCell; i++) {
	
	float rho_i = Rho[cell][i];

	for (int k = 0; k < 14; k++) {

	  int neighborCell = cell + cellOffsets[k];
	  if (neighborCell >= numCells) {
	    continue;
	  }

	  unsigned numParticlesInNeighborCell = X[neighborCell].size();
	  for (unsigned j = 0; j < numParticlesInNeighborCell; j++) {

	    if (k == 0 && Id[neighborCell][j] >= Id[cell][i]) {
	      continue;
	    }

	    float rho_j = Rho[neighborCell][j];
      	    float3 r = make_float3(X[cell][i] - X[neighborCell][j]);
	    float magr = length(r);
	    if (magr > h) {
	      continue;
	    }
	    if (magr > 0.0f) {
	      r = normalize(r);
	    }

	    // viscosity
	    float tmp = dynVisc*m*laplWviscosity(magr)*2.0f/(rho_i+rho_j);
	    float3 vdiff = make_float3(U[neighborCell][j] - U[cell][i]);

	    F[cell][i] += make_float4(vdiff*tmp);
	    F[neighborCell][j] += make_float4(vdiff*(-tmp));

	    // surface tension
	    float md = m*2.0f/(rho_i+rho_j);
	    float maggradcolor = gradWpoly6(magr);

	    r *= maggradcolor;
	    float4 st = make_float4(r.x, r.y, r.z, laplWpoly6(magr));

	    K[cell][i] += st*md;
	    st.x = -st.x;
	    st.y = -st.y;
	    st.z = -st.z;
	    K[neighborCell][j] += st*md;
	  }
	}
      }
    }
  }
}
#else
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

      	    float3 r = make_float3(ri.x-rj.x, ri.y-rj.y, ri.z-rj.z);
	    float magr = length(r);
	    if (magr > h) continue;

	    if (magr > 0.0f) {
	      r = normalize(r);
	    }

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

	    r.x *= maggradcolor;
	    r.y *= maggradcolor;
	    r.z *= maggradcolor;

	    ri.gx += r.x*md;
	    ri.gy += r.y*md;
	    ri.gz += r.z*md;
	    ri.lc += md*laplWpoly6(magr);

	    rj.gx += -r.x*md;
	    rj.gy += -r.y*md;
	    rj.gz += -r.z*md;
	    rj.lc += md*laplWpoly6(magr);
	  }
	}
      }
    }
  }
}
#endif//REMOVESTRUCT
#if REMOVESTRUCT
void finalizeOtherForces()
{
#pragma omp parallel for
  for (unsigned i = 0; i < F.size(); i++) {
    for (unsigned j = 0; j < F[i].size(); j++) {

      // gravity
      F[i][j] += make_float4(a_gravity*Rho[i][j]);

      // surface tension
      float3 gradcolor = make_float3(K[i][j]);
      if (length(gradcolor) > 0.0f) {
	gradcolor = normalize(gradcolor);
      }

      float tmp = -surfTension*K[i][j].w;
      F[i][j] += make_float4(tmp*gradcolor);
    }
  }
}
#else
void finalizeOtherForces()
{
#pragma omp parallel for
  for (unsigned i = 0; i < plists.size(); i++) {
    for (unsigned j = 0; j < plists[i].size(); j++) {

      // gravity
      plists[i][j].fx += a_gravity.x*plists[i][j].rho;
      plists[i][j].fy += a_gravity.y*plists[i][j].rho;
      plists[i][j].fz += a_gravity.z*plists[i][j].rho;

      // surface tension
      float3 gradcolor = make_float3(plists[i][j].gx,
				     plists[i][j].gy,
				     plists[i][j].gz);
      if (length(gradcolor) > 0.0f) {
	gradcolor = normalize(gradcolor);
      }

      float tmp = -surfTension*plists[i][j].lc;

      plists[i][j].fx += tmp*gradcolor.x;
      plists[i][j].fy += tmp*gradcolor.y;
      plists[i][j].fz += tmp*gradcolor.z;  
    }
  }
}
#endif
void computeOtherForcesComplete()
{
  computeOtherForces(0);
  computeOtherForces(1);
  finalizeOtherForces();
}
#if REMOVESTRUCT
void integrate(std::vector<float4>& particles, 
	       std::vector<float4>& velocities)
{
#pragma omp parallel for
  for (unsigned i = 0; i < F.size(); i++) {
    for (unsigned j = 0; j < F[i].size(); j++) {

      int idx = Id[i][j];
      float sc = timestep/Rho[i][j];

      velocities[idx] += F[i][j]*sc;
      particles[idx] += velocities[idx]*timestep;
    }
  }
}
#else
void integrate(std::vector<float4>& particles, 
	       std::vector<float4>& velocities)
{
#pragma omp parallel for
  for (unsigned i = 0; i < plists.size(); i++) {
    for (unsigned j = 0; j < plists[i].size(); j++) {

      int idx = plists[i][j].idx;

      float4 &particle = particles[idx];
      float4 &velocity = velocities[idx];
      float sc = timestep/plists[i][j].rho;

    
      velocity.x += plists[i][j].fx*sc;
      velocity.y += plists[i][j].fy*sc;
      velocity.z += plists[i][j].fz*sc;

      particle.x = particle.x + velocity.x*timestep;
      particle.y = particle.y + velocity.y*timestep;
      particle.z = particle.z + velocity.z*timestep;      
    }
  }
}
#endif//REMOVESTRUCT
void checkBoundaries(std::vector<float4>& particles, 
		     std::vector<float4>& velocities)
{
  // check particles on/behind the boundary------------------------------------
  unsigned numParticles = particles.size();
#pragma omp parallel for
  for (unsigned i = 0; i < numParticles; i++) {

    float4 &particle = particles[i];
    float4 &velocity = velocities[i];
    float damping = 1.0f;

    if (particle.x < 0.0f) {
      particle.x = 0.0f;
      velocity.x = -damping*velocity.x;
    }
    if (particle.x >  tankSize) {
      particle.x = tankSize;
      velocity.x = -damping*velocity.x;
    }

    if (particle.y < 0.0f) {
      particle.y = 0.0f;
      velocity.y = -damping*velocity.y;
    }
    if (particle.y >  tankSize) {
      particle.y = tankSize;
      velocity.y = -damping*velocity.y;
    }

    if (particle.z < 0.0f) {
      particle.z = 0.0f;
      velocity.z = -damping*velocity.z;
    }
    if (particle.z >  tankSize) {
      particle.z = tankSize;
      velocity.z = -damping*velocity.z;
    }
  }
}
#if PCISPH
void updateParticles(std::vector<float4>& particles, 
		     std::vector<float4>& velocities)
{
  assignParticlesToCells(particles, velocities);

}
#else
void updateParticles(std::vector<float4>& particles, 
		     std::vector<float4>& velocities)
{
  assignParticlesToCells(particles, velocities);

  computeDensityComplete();
  computePressureForcesComplete();
  computeOtherForcesComplete();
  integrate(particles, velocities);
  checkBoundaries(particles, velocities);
}
#endif
