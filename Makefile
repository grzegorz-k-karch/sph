CXX = g++
CXXFLAGS = -std=c++11 -fopenmp -Wall

INCLUDES = -Irender -I. -Icompute -I$(HOME)/NVIDIA_CUDA-6.5_Samples/common/inc/ -I/usr/local/cuda/include/
LIBDIRS = -Lrender/GLFW
LIBRARIES = -lglfw3 -lGL -lGLEW -lX11 -lXrandr -lXi -lXxf86vm

BIN = sph

.PHONY: all
all: $(BIN)

debug: CXXFLAGS += -DDEBUG -g
debug: $(BIN)

release: CXXFLAGS += -O3
release: $(BIN)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<
helper_glsl.o: render/helper_glsl.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<
trackball.o: render/trackball.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<
compute.o: compute/compute.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<
render.o: render/render.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<
$(BIN): main.o helper_glsl.o trackball.o compute.o render.o
	$(CXX) $(CXXFLAGS) $^ $(LIBDIRS) $(LIBRARIES) -o $@

.PHONY: clean
clean:
	rm -fv *.o *~ $(BIN)
