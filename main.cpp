/*
 * based on http://antongerdelan.net/opengl/hellotriangle.html
 * and http://www.opengl-tutorial.org/beginners-tutorials/tutorial-2-the-first-triangle/
 */
/*
 * parameters taken from
 * https://www8.cs.umu.se/kurser/5DV058/VT10/lectures/sphsurvivalkit.pdf
 * https://www8.cs.umu.se/kurser/5DV058/VT10/lectures/Lecture8.pdf
 * parameters adjusted according to the fluids_v1 program:
 * http://www.rchoetzlein.com/eng/graphics/fluids.htm
 */
/*
 * https://www.youtube.com/watch?v=SQPCXzqH610
 */
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "glm/glm.hpp"
#include "glm/core/type_vec3.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "helper_glsl.h"
#include "trackball.h"

GLFWwindow *window;
GLint windowWidth = 1200;
GLint windowHeight = 800;

GLfloat curquat[4];
GLfloat lastquat[4];


GLfloat modelTranslation[3] = { 0.0f, 0.0f, -3.5f };

double mousePosX;
double mousePosY;

GLuint vao = 0;
GLuint vaoParticles = 0;
GLuint vboParticles = 0;
GLuint program;

GLuint vaoSphere = 0;
GLuint vboSphereVertex = 0;
GLuint vboSphereNormal = 0;
GLuint vboSphereIndex = 0;
int numIndicesSphere;

GLuint vboPositions = 0;

int numParticles = 1500;
float *particles = 0;
float *velocities = 0;
// float k = 8.3144621f; // gas constant; here I simulate liquid
float rho0 = 600.0f;//FROM FLUIDS// 1000.0f; // [kg/m^3] - water
float dynVisc = 0.2f;//FROM FLUIDS// [kg/m/s]  (0.00089 [Pa*s]  - dynamic viscosity of water)
float soundSpeed = 1.0f; // should be 1500 [m/s] for water
float surfTension = 0.0000001197f;// surface tension coefficient of water against air at 25 degrees Celcius [dyn/cm] = 0.001 [N/m]
float tankSize = 0.2154f; // [m] side of a cube container with 10 liter of water

float a_gravity[3] = {0.0f, -10.0f, 0.0f};
float timestep = 0.005f; // 0.0001 [s] for speed of sound in water = 1500 [m/s]
float m = 0.00020543f;//FROM FLUIDS// 0.01 [m] for 1000 particles in 10L water with sound speed = 1500
float h = 0.01f;//FROM FLUIDS// 0.05848 [m] is the side of cube with average 20 particles if 1000 particles are in 10 liters of water
float h9 = 0.0f;
float h2 = 0.0f;
float h6 = 0.0f;

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

void updateParticles()
{
  float *f_total = (float*)malloc(numParticles*3*sizeof(float));
  for (int i = 0; i < numParticles*3; i++) {
    f_total[i] = 0.0f;
  }


  float avgDensity = 0.0f;
  float *densities = (float*)malloc(numParticles*sizeof(float));

  // compute density for each particle-----------------------------------------
#pragma omp parallel for
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

  glBindBuffer(GL_ARRAY_BUFFER, vboParticles);
  glBufferData(GL_ARRAY_BUFFER, numParticles*3*sizeof(float), particles, GL_STATIC_DRAW);
}

void error_callback(int error, const char* description)
{
  fputs(description, stderr);
}

void window_close_callback(GLFWwindow* window)
{
  if (particles != 0)
    free(particles);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
    glfwSetWindowShouldClose(window, GL_TRUE);
}

static void cursor_pos_callback(GLFWwindow* window, double x, double y)
{
  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
	
    float dy = (float)(y - mousePosY);
    float scaleZoom = fabs(modelTranslation[2]) > 0.1f ? fabs(modelTranslation[2]) : 0.1f;
    modelTranslation[2] += dy / 100.0f * scaleZoom;
  }
  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {

    trackball(lastquat, (2.0f*mousePosX - windowWidth) / windowWidth,
	      (windowHeight - 2.0f*mousePosY) / windowHeight,
	      (2.0f*x - windowWidth) / windowWidth,
	      (windowHeight - 2.0f*y) / windowHeight);
    add_quats(lastquat, curquat, curquat);
  }
  mousePosX = x;
  mousePosY = y;
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int bits)
{
  if (action == GLFW_RELEASE) {
    switch (button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      break;
    case GLFW_MOUSE_BUTTON_RIGHT:
      break;
    default:
      break;
    }
  }
}

static void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
  glViewport(0, 0, width, height);
  windowWidth = width;
  windowHeight = height;
}

void initParticles()
{
  particles = (float*)malloc(numParticles*3*sizeof(float));
  velocities = (float*)malloc(numParticles*3*sizeof(float));
  srand(time(0));

  for (int i = 0; i < numParticles; i++) {

    particles[i*3+0] = float(rand())/RAND_MAX*tankSize/2.0f;
    particles[i*3+1] = float(rand())/RAND_MAX*tankSize/2.0f;
    particles[i*3+2] = float(rand())/RAND_MAX*tankSize/2.0f;

    velocities[i*3+0] = 0.0f;
    velocities[i*3+1] = 0.0f;
    velocities[i*3+2] = 0.0f;
  }
}

void createSphere(float radius,
		  std::vector<float>& vertices, 
		  std::vector<unsigned int>& indices, 
		  std::vector<float>& normals)
{
  int stacks = 16;
  int slices = 16;

  for(int i = 0; i <= stacks; i++) {

    float lat0 = M_PI * (-0.5 + (float) (i) / stacks);
    float z0  = sin(lat0)*radius;
    float zr0 =  cos(lat0);
          
    for(int j = 0; j < slices; j++) {

      float lng = 2 * M_PI * (float) (j - 1) / slices;
      float x = cos(lng)*radius;
      float y = sin(lng)*radius;
      float v[4] = {x*zr0, y*zr0, z0, 1.0f};
      vertices.push_back(v[0]);
      vertices.push_back(v[1]);
      vertices.push_back(v[2]);
      vertices.push_back(v[3]);

      float n3[3] = {x * zr0, y * zr0, z0};
      vecNormalize(n3,n3);
      normals.push_back(n3[0]);
      normals.push_back(n3[1]);
      normals.push_back(n3[2]);
      normals.push_back(0.0f);
    }
  }
  for (int i = 0; i < stacks; i++) {
    for (int j = 0; j < slices; j++) {

      indices.push_back(i*slices+j);		   
      indices.push_back(i*slices+((j+1)%slices));	   
      indices.push_back((i+1)*slices+((j+1)%slices));
      indices.push_back((i+1)*slices+((j+1)%slices));
      indices.push_back((i+1)*slices+j);
      indices.push_back(i*slices+j);
    }
  }
}


void transform()
{
  float fnear = 0.01f;
  float ffar = 10.0f;
  float fov = 45;
  float aspect = (float)windowWidth / (float)windowHeight;
  glm::mat4 projMX, modelMX, viewMX;
  glm::mat4 modelInvTranspMX;

  projMX = glm::perspective(fov, aspect, fnear, ffar);
  viewMX = glm::mat4(1.0f);
  modelMX = glm::translate(glm::mat4(1.0f), glm::vec3(modelTranslation[0],
						      modelTranslation[1], 
						      modelTranslation[2]));
  modelInvTranspMX = glm::transpose(glm::inverse(modelMX));

  glm::mat4 m;
  build_rotmatrix(m, curquat);
  modelMX = glm::mat4(modelMX*m);

  glUniformMatrix4fv(glGetUniformLocation(program, "model_mat"),
		     1, GL_FALSE, glm::value_ptr(modelMX));
  glUniformMatrix4fv(glGetUniformLocation(program, "view_mat"),
		     1, GL_FALSE, glm::value_ptr(viewMX));
  glUniformMatrix4fv(glGetUniformLocation(program, "proj_mat"),
		     1, GL_FALSE, glm::value_ptr(projMX));
  glUniformMatrix4fv(glGetUniformLocation(program, "model_it_mat"),
		     1, GL_FALSE, glm::value_ptr(modelInvTranspMX));
}

void initGeometry()
{
  float vertices[] = {
    -1,-1,-1,
    1,-1,-1,
    -1,1,-1,
    1,1,-1,
    -1,-1,1,
    1,-1,1,
    -1,1,1,
    1,1,1
  };
  unsigned int vbo = 0;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, 24*sizeof(float), vertices, GL_STATIC_DRAW);

  unsigned int indices[] = {
    0,1,
    0,2,
    1,3,
    2,3,
    4,5,
    4,6,
    5,7,
    6,7,
    0,4,
    1,5,
    2,6,
    3,7
  };
  unsigned int ibo = 0;
  glGenBuffers(1, &ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, 24*sizeof(unsigned int),
  	       indices, GL_STATIC_DRAW);

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLubyte*)NULL);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

  // sphere for instancing

  float radius = 0.0025f;
  std::vector<float> Vertices;
  std::vector<float> Normals;  
  std::vector<unsigned int> Indices;
  
  Vertices.clear();
  Normals.clear();
  Indices.clear();
  createSphere(radius, Vertices, Indices, Normals);
  
  glGenVertexArrays(1, &vaoSphere);
  glBindVertexArray(vaoSphere);

  glGenBuffers(1, &vboSphereVertex);
  glBindBuffer(GL_ARRAY_BUFFER, vboSphereVertex);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float)*Vertices.size(), Vertices.data(), 
	       GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (GLubyte*)NULL);

  glGenBuffers(1, &vboSphereNormal);
  glBindBuffer(GL_ARRAY_BUFFER, vboSphereNormal);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float)*Normals.size(), Normals.data(), 
	       GL_STATIC_DRAW);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, (GLubyte*)NULL);

  glGenBuffers(1, &vboSphereIndex);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboSphereIndex);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*Indices.size(), 
	       Indices.data(), GL_STATIC_DRAW);

  numIndicesSphere = Indices.size();

  glGenBuffers(1, &vboPositions);
  glBindBuffer(GL_ARRAY_BUFFER, vboPositions);

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(float)*3, (GLubyte*)NULL);
  glVertexAttribDivisor(2, 1);

  glBindVertexArray(0);
}

unsigned int initShaderProgram()
{
  const GLchar *vertexShader[1] = { "./render/shaders/vs_basic.glsl" };
  const GLchar *fragmentShader[1] = { "./render/shaders/fs_simple.glsl" };
  GLuint sp = buildGLSLProgram(vertexShader, 1, fragmentShader, 1);

  return sp;
}

void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glUseProgram(program);

  transform();

  clock_t t = clock();

  updateParticles();

  t = clock() - t;
  std::cout << "elapsed time: " << float(t)/CLOCKS_PER_SEC << "s" << std::endl;

  glBindVertexArray(vaoSphere);
  glBindBuffer(GL_ARRAY_BUFFER, vboPositions);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float)*numParticles*3, particles, GL_DYNAMIC_DRAW);
  glDrawElementsInstanced(GL_TRIANGLES, numIndicesSphere, GL_UNSIGNED_INT, 0, numParticles);

  glfwPollEvents();

  glfwSwapBuffers(window);
}

void initOpenGL()
{
  glfwSetErrorCallback(error_callback);

  if (!glfwInit()) {
    printf("Failed to initialize GLFW\n");
    exit(EXIT_FAILURE);
  }

  window = glfwCreateWindow(windowWidth, windowHeight, "OpenGL", NULL, NULL);
  if (!window) {
    printf("Could not open window with GLFW\n");
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
  glfwMakeContextCurrent(window);

  // Callbacks
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetWindowCloseCallback(window, window_close_callback);
  glfwSetKeyCallback(window, key_callback);
  glfwSetCursorPosCallback(window, cursor_pos_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);

  // start GLEW extension handler
  glewInit();
  if (glewInit() != GLEW_OK) {
    printf("Failed to initialize GLEW\n");
    exit(EXIT_FAILURE);
  }

  glEnable(GL_DEPTH_TEST); // enable depth-testing

  trackball(curquat, 0.0f, 0.0f, 0.0f, 0.0f);

  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
}
//from http://stackoverflow.com/questions/11071116/i-got-omp-get-num-threads-always-return-1-in-gcc-works-in-icc
int omp_thread_count() 
{
  int n = 0;
#pragma omp parallel reduction(+:n)
  n += 1;
  return n;
}

int main()
{
  // std::cout << omp_thread_count() << ", " << omp_get_num_threads() << std::endl;

  h2 = h*h;
  h6 = h2*h2*h2;
  h9 = h*h2*h6;

  initOpenGL();

  initGeometry();
  initParticles();

  program = initShaderProgram();

  while (!glfwWindowShouldClose(window)) {
    display();
  }

  // close GL context and any other GLFW resources
  glfwTerminate();
  return 0;
}

