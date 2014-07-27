#include "helper_glsl.h"
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>

#define MAX_INFO_LOG_SIZE 2048

static std::string loadShaderText(const char* filename)
{
  std::ifstream file(filename, std::ifstream::in);
  std::string shaderCode = "";

  if (file.good()) {

    std::string line = "";
    while (getline(file, line))
      shaderCode += "\n" + line;

    shaderCode += '\0';
    //std::cout << shaderCode << std::endl;
  }
  else {
    std::cerr << "[helper_glsl:" << __LINE__ << "] " << "Could not open shader file: " << filename << std::endl;
  }
  file.close();
  return shaderCode;
}

static GLint validateProgram(GLuint program) 
{
  GLint success;
  GLchar infoLog[MAX_INFO_LOG_SIZE];

  // Check if linking succeeded
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(program, MAX_INFO_LOG_SIZE, NULL, infoLog);
    std::cerr << "[helper_glsl:" << __LINE__ << "] " << infoLog << std::endl;
  }
  // Validate program
  glGetProgramiv(program, GL_VALIDATE_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(program, MAX_INFO_LOG_SIZE, NULL, infoLog);
    std::cerr << "[helper_glsl:" << __LINE__ << "] " << infoLog << std::endl;
  }
  return success;
}

GLuint buildGLSLProgram(const GLchar** vertexShaderPath, GLint numVS,
                        const GLchar** fragmentShaderPath, GLint numFS)
{
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  GLuint program = glCreateProgram();
  GLint success;
  GLchar infoLog[MAX_INFO_LOG_SIZE];

  const GLchar **vertexShaderText = (const GLchar**)malloc(numVS*sizeof(GLchar*));
  const GLchar **fragmentShaderText = (const GLchar**)malloc(numFS*sizeof(GLchar*));
  std::vector<std::string> vertexShaderString(numVS);
  std::vector<std::string> fragmentShaderString(numFS);

  for (GLsizei i = 0; i < numVS; i++) {
    vertexShaderString[i] = loadShaderText(vertexShaderPath[i]);
    vertexShaderText[i] = vertexShaderString[i].c_str();
  }
  for (GLsizei i = 0; i < numFS; i++) {
    fragmentShaderString[i] = loadShaderText(fragmentShaderPath[i]);
    fragmentShaderText[i] = fragmentShaderString[i].c_str();
  }

  glShaderSource(vertexShader, numVS, (const GLchar**)vertexShaderText, NULL);
  glShaderSource(fragmentShader, numFS, (const GLchar**)fragmentShaderText, NULL);

  // Compile vertex shader and check if succeeded
  glCompileShader(vertexShader);
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, MAX_INFO_LOG_SIZE, NULL, infoLog);
    std::cerr << "[helper_glsl:" << __LINE__ << "] " << infoLog << std::endl;
    return 0;
  }
  // Compile fragment shader and check if succeeded
  glCompileShader(fragmentShader);
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, MAX_INFO_LOG_SIZE, NULL, infoLog);
    std::cerr << "[helper_glsl:" << __LINE__ << "] " << infoLog << std::endl;
    return 0;
  }

  // Attach shaders to program
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);

  // Link program and check if succeeded
  glLinkProgram(program);
  success = validateProgram(program);
  if (!success) {
    return 0;
  }

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  return program;
}

void deleteGLSLProgram(GLuint program)
{
  if (!glIsProgram(program)) {
    return;
  }

  const GLsizei maxNumShaders = 1024;
  GLsizei numReturnedShaders;
  GLuint shaders[maxNumShaders];

  glUseProgram(0);

  glGetAttachedShaders(program, maxNumShaders, &numReturnedShaders, shaders);
  for (GLsizei i = 0; i < numReturnedShaders; i++) {
    glDetachShader(program, shaders[i]);
    glDeleteShader(shaders[i]);
  }

  glDeleteProgram(program);
}
