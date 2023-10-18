#pragma once
#include <cassert>
#include <cstdlib>

//
// Structure to hold arrays of coordinates, masses and forces per particle
// Number of particles "n" is included too
// 
// Asserts that number of particles is greater than zero when allocating memory
//

struct Particles
{
    int n;                                  // Number of particles
	double *x, *y, *z;          // 3 dimension coordinates
    double *m;                  // mass of each particle
	double *fx, *fy, *fz;       // Forces in each Coordinate
	
    // empty Constructor with initialization list (n = -1)
	Particles() : n(-1) {}
    // Constructor of n particles.
	Particles(int n) : n(n) { allocate(n); }
	void allocate(int n){
		assert(n > 0);
		
		this->n = n;
		x  = (double*)aligned_alloc(64,n*sizeof(double));
		y  = (double*)aligned_alloc(64,n*sizeof(double));
		z  = (double*)aligned_alloc(64,n*sizeof(double));
		m  = (double*)aligned_alloc(64,n*sizeof(double));
		
		fx = new double[n];
		fy = new double[n];
		fz = new double[n];
	}
	
    // Destructor of structure
	~Particles()
	{
		if (n <= 0) return;
		
		delete[] x;
		delete[] y;
		delete[] z;
		delete[] m;
		delete[] fx;
		delete[] fy;
		delete[] fz;
	}	
};
