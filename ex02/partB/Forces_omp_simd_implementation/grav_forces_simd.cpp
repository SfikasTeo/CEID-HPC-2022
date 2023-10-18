#include <x86intrin.h>
#include <cmath>

#include "particles.h"
#include "utils.h"

//
// Compute the gravitational forces in the system of particles
// Symmetry of the forces is NOT exploited
//

extern __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c);

inline void seriallyComputatingRemainingParticles(int currentParticle, int remaining, Particles& particles){
  	constexpr double G = 6.67408e-11;
    double xr, yr, zr, tmp, magnitude;
    for (int j=particles.n-remaining; j<particles.n; j++){
		if ( currentParticle != j ){
            xr = particles.x[currentParticle]-particles.x[j];
            yr = particles.y[currentParticle]-particles.y[j];
            zr = particles.z[currentParticle]-particles.z[j];

			tmp = xr*xr + yr*yr + zr*zr;
			magnitude = G * particles.m[currentParticle] * particles.m[j] / pow(tmp, 1.5);

			particles.fx[currentParticle] += xr * magnitude;
			particles.fy[currentParticle] += yr * magnitude;
			particles.fz[currentParticle] += zr * magnitude;
		}
    }
}

inline double hsum(const __m256d vector){
  return vector[0] + vector[1] + vector[2] + vector[3];
}

void computeGravitationalForces(Particles& particles)
{
	const __m256d G = _mm256_set1_pd(6.67408e-11);
    __m256d xi, yi, zi, xj, yj, zj, mi, mj;
    __m256d xr, yr, zr;
    __m256d aux1, aux2, aux3, aux4;
    int i, j, remaining;
 
    // For each particle initialize and
    // compute gravitational forces
	for ( i=0; i<particles.n; i++)
	{
        // The dynamic memory initialzation
        // with the use of new double[n]
        // Does NOT automatically initialize
        // values to 0
		particles.fx[i] = 0;
		particles.fy[i] = 0;
		particles.fz[i] = 0;
        
        xi = _mm256_set1_pd( particles.x[i] );
        yi = _mm256_set1_pd( particles.y[i] );
        zi = _mm256_set1_pd( particles.z[i] );
        mi = _mm256_set1_pd( particles.m[i] );

        // Calculate the gracitational forces that 
        // affect the current particle from every other.
		for ( j=0; j+4 <= particles.n; j+=4) {
            xj = _mm256_load_pd( &particles.x[j] );
            yj = _mm256_load_pd( &particles.y[j] );
            zj = _mm256_load_pd( &particles.z[j] );
            mj = _mm256_load_pd( &particles.m[j] );
			            
            xr = _mm256_sub_pd(xi,xj);
            yr = _mm256_sub_pd(yi,yj);
            zr = _mm256_sub_pd(zi,zj);

            // IMPORTANT => performance hit => operational chains => I do not know 
            // how the cpu pipelining and the compiler will optimize the operation.
            // Furthermore on AMD CPU systems the support of _mm256_fmadd_pd 
            // instruction and FMA4 is debatable (1). Working alternative:
            // _mm256_add_pd(_mm256_mul_pd(aXX, bYY), cZZ)
            // 1: https://www.techpowerup.com/248560/amd-zen-does-support-fma4-just-not-exposed

                // Option 1: Will this pipeline correctly ?
                // xr*xr + yr*yr + zr*zr
			    aux4 = _mm256_fmadd_pd(xr,xr, _mm256_fmadd_pd(yr,yr, _mm256_mul_pd(zr,zr)));

                // Option 2: Most Certainly => Latency hit not throughput.
                /*aux1 = _mm256_mul_pd(xr,xr);  aux2 = _mm256_mul_pd(yr,yr);
                aux3 = _mm256_add_pd(aux1,aux2);aux4 = _mm256_mul_pd(zr,zr);
                aux4 = _mm256_add_pd(aux4,aux3);*/

            // The two options have to be benchmarked for large datasets
            // in order to spot any actual difference in performance.
            
            // aux^1.5 = aux^(3/2) = (aux^3)^1/2 = temp^(1/2);
            aux4 = _mm256_sqrt_pd( _mm256_mul_pd(aux4,_mm256_mul_pd(aux4,aux4)));
            
            aux1 = _mm256_mul_pd(mi,mj);
            aux2 = _mm256_mul_pd(aux1,G);
			aux3 = _mm256_div_pd( aux2, aux4);
            
            // If for one value of the __m256d vector register => i==j 
            // => magnutude = 0 => particle.fi==j += 0 
            if (j <= i && i < j+4){
                const __m256d is = _mm256_set1_pd(i);
                const __m256d js = _mm256_set_pd( j+3.0, j+2.0, j+1.0, j);
                const __m256d mask = _mm256_cmp_pd(is, js, _CMP_NEQ_OQ);
                aux3 = _mm256_and_pd( aux3, mask);
            }

			particles.fx[i] += hsum(_mm256_mul_pd(xr, aux3));
			particles.fy[i] += hsum(_mm256_mul_pd(yr, aux3));
			particles.fz[i] += hsum(_mm256_mul_pd(zr, aux3));
        }
        
        remaining = particles.n - (int)particles.n/4 * 4;
        switch ( remaining ) {
            case 3: seriallyComputatingRemainingParticles(i,remaining,particles); break;
            case 2: seriallyComputatingRemainingParticles(i,remaining,particles); break;
            case 1: seriallyComputatingRemainingParticles(i,remaining,particles); break;
        }
	}
}

int main()
{
	testAll(computeGravitationalForces);
	return 0;
}
