#include "particles.h"
#include "utils.h"
#include <cmath>

//
// Compute the gravitational forces in the system of particles
// Symmetry of the forces is NOT exploited
//

void computeGravitationalForces(Particles& particles)
{
	constexpr double G = 6.67408e-11;

    // For each particle initialize forces
	for (int i=0; i<particles.n; i++)
	{
		// The dynamic memory initialzation
        // with the use of new double[n]
        // Does NOT automatically initialize
        // values to 0
        particles.fx[i] = 0;
		particles.fy[i] = 0;
		particles.fz[i] = 0;

        // Calculate the gracitational forces that 
        // affect the current particle from every other.
		for (int j=0; j<particles.n; j++)
			if (i!=j)
			{
				double tmp = pow(particles.x[i]-particles.x[j], 2.0) +
							 pow(particles.y[i]-particles.y[j], 2.0) +
							 pow(particles.z[i]-particles.z[j], 2.0);

				double magnitude = G * particles.m[i] * particles.m[j] / pow(tmp, 1.5);

				particles.fx[i] += (particles.x[i]-particles.x[j]) * magnitude;
				particles.fy[i] += (particles.y[i]-particles.y[j]) * magnitude;
				particles.fz[i] += (particles.z[i]-particles.z[j]) * magnitude;
			}
	}
}

int main()
{
	testAll(computeGravitationalForces);
	return 0;
}
