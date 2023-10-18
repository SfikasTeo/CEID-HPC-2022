#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#include "func.c"

#define MAX_NNB	256
#ifndef MANUAL
	#define PROBDIM 16
	#define NNBS 32
	#define TRAINELEMS 1048576
	#define QUERYELEMS 1024
	#define LOW  0
	#define HIGH 2
#endif

double compute_euclidean_distance( const double *restrict v, const double *restrict w, int n)
{
	int i;
	double s = 0.0;
	for (i = 0; i < n; i++) {
		s += pow(v[i]-w[i],2);
	}
	return sqrt(s);
}

double find_knn_value(	const double *restrict xquery, const double *restrict ytr, \
						const double *restrict xtr, const int prob_dim, const int knn )
{
	int i, nn_x[MAX_NNB] = {-1}, max_i=0;
	double nn_d[MAX_NNB], new_d, max_d=1e99;

	for (i = 0; i < knn; i++) {
		nn_d[i] = 1e99-i;
	}

	// Brute force -> find the knn 
	for (i = 0; i < TRAINELEMS; i++) {
		new_d = compute_euclidean_distance(xquery, &xtr[i*prob_dim], prob_dim);
		if (new_d < max_d) {
			nn_x[max_i] = i;
			nn_d[max_i] = new_d;
			max_d = compute_max_position(nn_d, knn, &max_i);
		}
	}
		
	// predict value
	double sum = 0.0;
	for (int i = 0; i < knn; i++)
		sum += ytr[nn_x[i]];

	return sum/knn;
}

int main(int argc, char *argv[])
{
	
	if (argc < 4)
	{
		printf("usage: %s <trainfile> <queryfile> <threads>\n", argv[0]);
		exit(1);
	}
	FILE *fpin = NULL;
	char *trainfile = argv[1];
	char *queryfile = argv[2];
	int threads = atoi(argv[3]);

	// Allocate memory to the CPU 
	// Allocate memory space for the Training Data.
	static double xtr[TRAINELEMS*PROBDIM];
	static double ytr[TRAINELEMS];

	// Allocate memory space for the Query Data.
	double xquery[QUERYELEMS*PROBDIM];
	double yquery[QUERYELEMS];

	// Read Training Data to Host.
	fpin = open_traindata(trainfile);
	
	for (int i=0;i<TRAINELEMS;i++) {
		for (int k = 0; k < PROBDIM; k++)
			xtr[i*PROBDIM+k] = read_nextnum(fpin);
			
			#if defined(SURROGATES)
				ytr[i] = read_nextnum(fpin);
			#else
				ytr[i] = 0;
				float temp = read_nextnum(fpin);
			#endif
	}
	fclose(fpin);

	// Read Query Data to Host.
	fpin = open_querydata(queryfile);
	for(int i = 0; i< QUERYELEMS; i++){
		for(int j = 0; j< PROBDIM; j++){
				xquery[i*PROBDIM + j] = read_nextnum(fpin);
			}
			yquery[i] = read_nextnum(fpin);
	}
	fclose(fpin);

	double t_sum=0.0;
	double sse=0.0;
	double err_sum=0.0;

	omp_set_num_threads(threads);
	omp_set_dynamic(0);

	double yp;
	double t_start = gettime();
	#pragma omp parallel reduction(+:sse) reduction(+:err_sum)
	{
		#pragma omp for
		for ( int i=0; i<QUERYELEMS; i++) {
			// Compute the yp and aproximate query value.
			yp = find_knn_value( &xquery[i*PROBDIM], ytr, xtr, PROBDIM, NNBS );
			// Calculate Error
			sse += (yquery[i]-yp)*(yquery[i]-yp);
			err_sum +=  100.0*fabs((yp-yquery[i])/yquery[i]);
		} 
	} // End of parallel region
	double t_end = gettime();

	double mse = sse/QUERYELEMS;
	double ymean = compute_mean(yquery, QUERYELEMS);
	double var   = compute_var(yquery, QUERYELEMS, ymean);

	t_sum = (t_end-t_start)*1000.0;		// convert to ms
		
	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum/QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", 1-(mse/var));
	printf("Total time = %lf ms\n", t_sum);
	printf("Average time/query = %lf ms\n", t_sum/QUERYELEMS);
	
	return 0;
}
