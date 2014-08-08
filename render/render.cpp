#include "render.h"
#include "compute.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "glm/glm.hpp"
#include "glm/core/type_vec3.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "helper_glsl.h"
#include "trackball.h"

#include <vector>
#include <iostream> 

GLFWwindow *window;
GLint windowWidth = 1200;
GLint windowHeight = 800;

GLfloat curquat[4];
GLfloat lastquat[4];


GLfloat modelTranslation[3] = { 0.0f, 0.0f, -1.0f };

double mousePosX;
double mousePosY;

GLuint vaoParticles = 0;
GLuint vboParticles = 0;
GLuint program;

GLuint vaoSphere = 0;
GLuint vboSphereVertex = 0;
GLuint vboSphereNormal = 0;
GLuint vboSphereIndex = 0;
int numIndicesSphere;

GLuint vboPositions = 0;

void terminateOpenGL()
{
  glfwTerminate();
}

void error_callback(int error, const char* description)
{
  fputs(description, stderr);
}

void window_close_callback(GLFWwindow* window)
{

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

void createSphere(float radius,
		  std::vector<float>& vertices, 
		  std::vector<unsigned int>& indices, 
		  std::vector<float>& normals)
{
  int stacks = 64;
  int slices = 64;

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
  // sphere for instancing

  float radius = 0.002f;
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

int display(float* particles, int numParticles)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glUseProgram(program);

  transform();

  glBindBuffer(GL_ARRAY_BUFFER, vboParticles);
  glBufferData(GL_ARRAY_BUFFER, numParticles*3*sizeof(float), particles, GL_STATIC_DRAW);

  glBindVertexArray(vaoSphere);
  glBindBuffer(GL_ARRAY_BUFFER, vboPositions);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float)*numParticles*3, particles, GL_DYNAMIC_DRAW);
  glDrawElementsInstanced(GL_TRIANGLES, numIndicesSphere, GL_UNSIGNED_INT, 0, numParticles);

  glfwPollEvents();

  glfwSwapBuffers(window);

  return glfwWindowShouldClose(window);
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

  program = initShaderProgram();
}
