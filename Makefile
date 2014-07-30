CXX = g++
CXXFLAGS = -std=c++11 -fopenmp

INCLUDES = -Irender -I.
LIBDIRS = -Lrender/GLFW
LIBRARIES = -lglfw3 -lGL -lGLEW -lX11 -lXrandr -lXi -lXxf86vm

BIN = sph

.PHONY: all
all: $(BIN)

debug: CXX += -DDEBUG -g
debug: $(BIN)

release: CXX += -O3
release: $(BIN)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<
helper_glsl.o: render/helper_glsl.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ -c $<
trackball.o: render/trackball.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<
$(BIN): main.o helper_glsl.o trackball.o
	$(CXX) $(CXXFLAGS) $^ $(LIBDIRS) $(LIBRARIES) -o $@

.PHONY: clean
clean:
	rm -f *.o *~ $(BIN)
