#ifndef HELPER_GLSL_H
#define HELPER_GLSL_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>

GLuint buildGLSLProgram(const GLchar** vertexShaderPath, GLint numVS,
                        const GLchar** fragmentShaderPath, GLint numFS);

void deleteGLSLProgram(GLuint program);

#endif//HELPER_GLSL_H
