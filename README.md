sph
===

fluid simulation based on smoothed particle hydrodynamics

graphics code based on:
 * http://antongerdelan.net/opengl/hellotriangle.html
 * http://www.opengl-tutorial.org/beginners-tutorials/tutorial-2-the-first-triangle/

fluid parameters taken from
 * https://www8.cs.umu.se/kurser/5DV058/VT10/lectures/sphsurvivalkit.pdf
 * https://www8.cs.umu.se/kurser/5DV058/VT10/lectures/Lecture8.pdf

parameters adjusted according to the fluids_v1 program:
 * http://www.rchoetzlein.com/eng/graphics/fluids.htm

optimization based on
https://software.intel.com/en-us/articles/fluid-simulation-for-video-games-part-15
and probably on
http://www.jontse.com/courses/files/cornell/cs5220/project02_sph.pdf


further materials:
 * https://www.youtube.com/watch?v=SQPCXzqH610
 * http://image.diku.dk/projects/media/kelager.06.pdf
 * Masterthesis in Informatik,Partikelbasierte Echtzeit-Fluidsimulation, Realtime particle-based fluid simulation,Stefan Auer

PARAMETERS
tank size		0.2154f		([m] side of a cube container with 10 liter of water)
speed of sound		1.0		(should be 1500 [m/s] for water)
particle mass 		0.00020543 	(taken from fluids_v1; should be m = 0.01 [kg] for 1000 particles in 10L water with sound speed = 1500 [m/s])
water density 		600.0 		(taken from fluids_v1; should be 1000.0 [kg/m^3] for water)
dynamic viscosity	0.2		(taken from fluids_v1; should be 0.00089 [kg/m/s or Pa*s] for dynamic viscosity of water)
gravity 		[0.0 -10.0 0.0]	(gravity vector [m/s^2])
time step		0.005	   	(should be 0.0001 [s] for speed of sound in water = 1500 [m/s])
smoothing length	0.01		(taken from fluids_v1; radius of kernel support; should be 0.05848 [m] - it is the side of cube with average 20 particles if 1000 particles are in 10 liters of water)


[NOT USED] gas constant k = 8.3144621f; // gas constant; here I simulate liquid
[NOT USED] surface tension = 0.0728f;// surface tension coefficient of water against air at 20 degrees Celcius [dyn/cm] = 0.001 [N/m]
