#include <time.h>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "compute.h"

float soundSpeed = 1.0f; // should be 1500 [m/s] for water
float m = 0.00020543f;//FROM FLUIDS// 0.01 [m] for 1000 particles in 10L water with sound speed = 1500
float rho0 = 600.0f;//FROM FLUIDS// 1000.0f; // [kg/m^3] - water
float dynVisc = 0.2f;//FROM FLUIDS// [kg/m/s]  (0.00089 [Pa*s]  - dynamic viscosity of water)
float a_gravity[3] = {0.0f, -10.0f, 0.0f};
float timestep = 0.005f; // 0.0001 [s] for speed of sound in water = 1500 [m/s]
float tankSize = 0.2154f; // [m] side of a cube container with 10 liter of water

float h = 0.01f;//FROM FLUIDS// 0.05848 [m] is the side of cube with average 20 particles if 1000 particles are in 10 liters of water
float h9 = 0.0f;
float h2 = 0.0f;
float h6 = 0.0f;


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


void initParticles(float** particles, float** velocities, 
		   int numParticles, float scale[3], float offset[3])
{
  h2 = h*h;
  h6 = h2*h2*h2;
  h9 = h*h2*h6;

  *particles = new float[numParticles*3];
  *velocities = new float[numParticles*3];
  srand(time(0));

  for (int i = 0; i < numParticles; i++) {

    (*particles)[i*3+0] = float(rand())/RAND_MAX*scale[0] + offset[0];
    (*particles)[i*3+1] = float(rand())/RAND_MAX*scale[1] + offset[1];
    (*particles)[i*3+2] = float(rand())/RAND_MAX*scale[2] + offset[2];

    (*velocities)[i*3+0] = 0.0f;
    (*velocities)[i*3+1] = 0.0f;
    (*velocities)[i*3+2] = 0.0f;
  }
}

void updateParticles(float* particles, float* velocities, int numParticles)
{
  float *f_total = (float*)malloc(numParticles*3*sizeof(float));
  for (int i = 0; i < numParticles*3; i++) {
    f_total[i] = 0.0f;
  }

  float avgDensity = 0.0f;
  float *densities = (float*)malloc(numParticles*sizeof(float));

  // compute density for each particle-----------------------------------------
#pragma omp parallel for reduction(+:avgDensity)
  for (int i = 0; i < numParticles; i++) {

    float rho = 0;
    float *ri = &particles[i*3];

    for (int j = 0; j < numParticles; j++) {

      float *rj = &particles[j*3];
      float r[3] = {(ri[0]-rj[0]), 
		    (ri[1]-rj[1]), 
		    (ri[2]-rj[2])};
      float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
      float magr = std::sqrt(r2);

      rho += m*Wpoly6(magr);
    }
    densities[i] = rho;
    avgDensity += rho;
  }

  std::cout << "avgDensity = " << avgDensity/numParticles << std::endl;

  // pressure-----------------------------------------------------------------
  float c = soundSpeed*soundSpeed;
#pragma omp parallel for
  for (int i = 0; i < numParticles; i++) {

    float *ri = &particles[i*3];
    float pi = c*(densities[i]-rho0);
    float fpressure[3] = {0.0f, 0.0f, 0.0f};

    for (int j = 0; j < numParticles; j++) {

      float *rj = &particles[j*3];
      float pj = c*(densities[j]-rho0);
      
      float r[3] = {(ri[0]-rj[0]), 
		    (ri[1]-rj[1]), 
		    (ri[2]-rj[2])};
      float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
      float mfp = -m*(pi+pj)/(2.0f*densities[j])*gradWspiky(magr);
      // FROM FLUIDS: (does not work well)
      // float mfp = -m*(pi+pj)/(densities[i]*densities[j])*gradWspiky(magr);

      vecNormalize(r,r);

      fpressure[0] += mfp*r[0];
      fpressure[1] += mfp*r[1];
      fpressure[2] += mfp*r[2];
    }

    f_total[i*3+0] += fpressure[0];
    f_total[i*3+1] += fpressure[1];
    f_total[i*3+2] += fpressure[2];
  }

  // visocity------------------------------------------------------------------
#pragma omp parallel for
  for (int i = 0; i < numParticles; i++) {

    float *ri = &particles[i*3];
    float *vi = &velocities[i*3];
    float fviscosity[3] = {0.0f, 0.0f, 0.0f};

    for (int j = 0; j < numParticles; j++) {

      float *rj = &particles[j*3];
      float *vj = &velocities[j*3];
      float vdiff[3] = {vj[0]-vi[0], vj[1]-vi[1], vj[2]-vi[2]};
      float r[3] = {(ri[0]-rj[0]), 
		    (ri[1]-rj[1]), 
		    (ri[2]-rj[2])};
      float magr = std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);

      fviscosity[0] += dynVisc*m*vdiff[0]/densities[j]*laplWviscosity(magr);
      fviscosity[1] += dynVisc*m*vdiff[1]/densities[j]*laplWviscosity(magr);
      fviscosity[2] += dynVisc*m*vdiff[2]/densities[j]*laplWviscosity(magr);

      // // FROM FLUIDS
      // fviscosity[0] += dynVisc*m*vdiff[0]/(densities[i]*densities[j])*laplWviscosity(magr);
      // fviscosity[1] += dynVisc*m*vdiff[1]/(densities[i]*densities[j])*laplWviscosity(magr);
      // fviscosity[2] += dynVisc*m*vdiff[2]/(densities[i]*densities[j])*laplWviscosity(magr);
    }
    
    f_total[i*3+0] += fviscosity[0];
    f_total[i*3+1] += fviscosity[1];
    f_total[i*3+2] += fviscosity[2];
  }

  // gravity-------------------------------------------------------------------
#pragma omp parallel for
  for (int i = 0; i < numParticles; i++) {

    f_total[i*3+0] += a_gravity[0]*densities[i];
    f_total[i*3+1] += a_gravity[1]*densities[i];
    f_total[i*3+2] += a_gravity[2]*densities[i];    
  }

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
