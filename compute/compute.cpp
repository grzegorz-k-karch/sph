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

vector<vector<float4>>    X; // Position
vector<vector<float4>>    U; // Velocity
vector<vector<unsigned>> Id; // Index
vector<vector<float>>   Rho; // Denity
vector<vector<float4>>    F; // Force
vector<vector<float4>>    K; // Surface tension
vector<int> cellSizes;
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

template <class T>
void resetArray(vector<T>& array)
{
  array.resize(cellSizes.size());
  for (auto i = 0; i < cellSizes.size(); i++) {
    array[i].clear();
    array[i].resize(cellSizes[i]);
  }
}

void assignParticlesToCells(vector<float4>& particles, 
			    vector<float4>& velocities,
			    vector<vector<float4>>& x,
			    vector<vector<float4>>& u,
			    vector<vector<unsigned>>& id)
{
  const float cellSize = h*1.01f;
  const int gridSize = std::ceil(tankSize/cellSize);
  const int numCells = gridSize*gridSize*gridSize;

  x.resize(numCells);
  u.resize(numCells);
  id.resize(numCells);

  for (int i = 0; i < numCells; i++) {
    x[i].clear();
    u[i].clear();
    id[i].clear();
  }

  const unsigned numParticles = particles.size();

  for (unsigned i = 0; i < numParticles; i++) {

    float4 pos = particles[i];
    int cellCoords[3] = {int(pos.x/cellSize), 
			 int(pos.y/cellSize), 
			 int(pos.z/cellSize)};
    int cellId = cellCoords[0] + 
      cellCoords[1]*gridSize + 
      cellCoords[2]*gridSize*gridSize;

    x[cellId].push_back(pos);
    u[cellId].push_back(velocities[i]);
    id[cellId].push_back(i);
  }

  cellSizes.resize(numCells);
  for (int i = 0; i < numCells; i++) {
    cellSizes[i] = x[i].size();
  }
}

void computeDensity(int z0,
		    vector<vector<float4>>& x_arg,
		    vector<vector<float>>& rho_arg)
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
      unsigned numParticlesInCell = x_arg[cell].size();
      for (unsigned i = 0; i < numParticlesInCell; i++) {

	float densityi = 0.0f;

	rho_arg[cell][i] += own_rho;

	for (int k = 0; k < 14; k++) {

	  int neighborCell = cell + cellOffsets[k];
	  if (neighborCell >= numCells) {
	    continue;
	  }

	  unsigned numParticlesInNeighborCell = x_arg[neighborCell].size();
	  for (unsigned j = 0; j < numParticlesInNeighborCell; j++) {

	    if (k == 0 && Id[neighborCell][j] >= Id[cell][i]) {
	      continue;
	    }
	    float4 r = x_arg[cell][i] - x_arg[neighborCell][j];
	    float r2 = r.x*r.x+r.y*r.y+r.z*r.z;
	    float magr = std::sqrt(r2);
	    if (magr > h) {
	      continue;
	    }
	    float rho = m*Wpoly6(magr);

	    densityi += rho;
	    rho_arg[neighborCell][j] += rho;
	  }
	}
	rho_arg[cell][i] += densityi;
      }
    }    
  }
}

void computeDensityComplete(vector<vector<float4>>& x_arg,
			    vector<vector<float>>& rho_arg)
{
  computeDensity(0, x_arg, rho_arg);
  computeDensity(1, x_arg, rho_arg);
}

void computePressureForces(int z0, 
			   vector<vector<float4>>& x_arg,
			   vector<vector<float>>& rho_arg,
			   vector<vector<float4>>& f_arg)
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
      unsigned numParticlesInCell = x_arg[cell].size();
      for (unsigned i = 0; i < numParticlesInCell; i++) {
	
	float rho_i = rho_arg[cell][i];
	float pi = cs*(rho_i-rho0);

	for (int k = 0; k < 14; k++) {

	  int neighborCell = cell + cellOffsets[k];
	  if (neighborCell >= numCells) {
	    continue;
	  }

	  unsigned numParticlesInNeighborCell = x_arg[neighborCell].size();
	  for (unsigned j = 0; j < numParticlesInNeighborCell; j++) {

	    if (k == 0 && Id[neighborCell][j] >= Id[cell][i]) {
	      continue;
	    }

	    float rho_j = rho_arg[neighborCell][j];
	    float pj = cs*(rho_j-rho0);
      	    float3 r = make_float3(x_arg[cell][i] - x_arg[neighborCell][j]);
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
	    f_arg[cell][i] += make_float4(mfp*r);
	    f_arg[neighborCell][j] += make_float4(-mfp*r);
	  }
	}
      }
    }
  }
}

void computePressureForcesComplete(vector<vector<float4>>& x_arg,
				   vector<vector<float>>& rho_arg,
				   vector<vector<float4>>& f_arg)
{
  computePressureForces(0, x_arg, rho_arg, f_arg);
  computePressureForces(1, x_arg, rho_arg, f_arg);
}

void computeOtherForces(int z0,
			vector<vector<float4>>& x_arg,
			vector<vector<float>>& rho_arg,
			vector<vector<float4>>& f_arg,
			vector<vector<float4>>& k_arg)
{
  const float cellSize = h*1.01f;
  const int gridSize = std::ceil(tankSize/cellSize);
  const int numCells = gridSize*gridSize*gridSize;
  const int numCellsInSlab = gridSize*gridSize;

#pragma omp parallel for 
  for (int z = z0; z < gridSize; z+=2) {    
    for (int s = 0; s < numCellsInSlab; s++) {      

      int cell = s + z*numCellsInSlab;
      unsigned numParticlesInCell = x_arg[cell].size();
      for (unsigned i = 0; i < numParticlesInCell; i++) {
	
	float rho_i = rho_arg[cell][i];

	for (int k = 0; k < 14; k++) {

	  int neighborCell = cell + cellOffsets[k];
	  if (neighborCell >= numCells) {
	    continue;
	  }

	  unsigned numParticlesInNeighborCell = x_arg[neighborCell].size();
	  for (unsigned j = 0; j < numParticlesInNeighborCell; j++) {

	    if (k == 0 && Id[neighborCell][j] >= Id[cell][i]) {
	      continue;
	    }

	    float rho_j = rho_arg[neighborCell][j];
      	    float3 r = make_float3(x_arg[cell][i] - x_arg[neighborCell][j]);
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

	    f_arg[cell][i] += make_float4(vdiff*tmp);
	    f_arg[neighborCell][j] += make_float4(vdiff*(-tmp));

	    // surface tension
	    float md = m*2.0f/(rho_i+rho_j);
	    float maggradcolor = gradWpoly6(magr);

	    r *= maggradcolor;
	    float4 st = make_float4(r.x, r.y, r.z, laplWpoly6(magr));

	    k_arg[cell][i] += st*md;
	    st.x = -st.x;
	    st.y = -st.y;
	    st.z = -st.z;
	    k_arg[neighborCell][j] += st*md;
	  }
	}
      }
    }
  }
}

void finalizeOtherForces(vector<vector<float>>& rho_arg,
			 vector<vector<float4>>& f_arg,
			 vector<vector<float4>>& k_arg)
{
#pragma omp parallel for
  for (unsigned i = 0; i < f_arg.size(); i++) {
    for (unsigned j = 0; j < f_arg[i].size(); j++) {

      // gravity
      f_arg[i][j] += make_float4(a_gravity*rho_arg[i][j]);

      // surface tension
      float3 gradcolor = make_float3(k_arg[i][j]);
      if (length(gradcolor) > 0.0f) {
	gradcolor = normalize(gradcolor);
      }

      float tmp = -surfTension*k_arg[i][j].w;
      f_arg[i][j] += make_float4(tmp*gradcolor);
    }
  }
}

void computeOtherForcesComplete(vector<vector<float4>>& x_arg,
				vector<vector<float>>& rho_arg,
				vector<vector<float4>>& f_arg,
				vector<vector<float4>>& k_arg)
{
  computeOtherForces(0, x_arg, rho_arg, f_arg, k_arg);
  computeOtherForces(1, x_arg, rho_arg, f_arg, k_arg);
  finalizeOtherForces(rho_arg, f_arg, k_arg);
}

void integrate(std::vector<float4>& particles, 
	       std::vector<float4>& velocities,
	       vector<vector<float4>>& f_arg)
{
#pragma omp parallel for
  for (unsigned i = 0; i < f_arg.size(); i++) {
    for (unsigned j = 0; j < f_arg[i].size(); j++) {

      int idx = Id[i][j];
      float sc = timestep/Rho[i][j];

      velocities[idx] += f_arg[i][j]*sc;
      particles[idx] += velocities[idx]*timestep;
    }
  }
}

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
  assignParticlesToCells(particles, velocities, X, U, Id);
  resetArray(Rho);
  resetArray(F);
  resetArray(K);

  computeDensityComplete(X, Rho);
  computePressureForcesComplete(X, Rho, F);
  computeOtherForcesComplete(X, Rho, F, K);
  integrate(particles, velocities, F);
  checkBoundaries(particles, velocities);
}
#endif
