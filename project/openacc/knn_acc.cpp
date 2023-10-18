#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

#include "func.c"
#include "timer.hpp"

#define MAX_NNB	256
#ifndef MANUAL
	#define PROBDIM 16
	#define NNBS 32
	#define TRAINELEMS 1048576
	#define QUERYELEMS 1024
	#define LOW  0
	#define HIGH 2
#endif

void find_dist_kernel( 	const float * __restrict__ xtr, const float * __restrict__ xquery, const int train, \
						const int prob_dim, const int knn, float * __restrict__ computed_distance )
{
	#pragma acc data copyout(computed_distance[0:TRAINELEMS]) deviceptr(xtr,xquery)
	{
		#pragma acc parallel num_gangs(TRAINELEMS/128) vector_length(128)
		{
			#pragma acc loop gang vector independent
			for (int i=0; i < TRAINELEMS; i++){
				#pragma acc cache(xquery[0:PROBDIM])
				
				float s = 0.0;
				#pragma acc loop seq
				for ( int j = 0; j < PROBDIM; j++) {
					s += (xquery[j] - xtr[i*PROBDIM+j])*(xquery[j] - xtr[i*PROBDIM+j]);
				}
				s = sqrt(s);
				computed_distance[i] = s;
			}
		}
	}
}
	
float find_knn_value( 	const float * __restrict__ xquery, const int prob_dim, const int knn,	const float * __restrict__ xtr, \
						const float * __restrict__ ytr, float * __restrict__ computed_distance )
{
	// initialize pairs of index and distance //
	int nn_x[MAX_NNB] = {-1}, max_i=0;
	float nn_d[MAX_NNB], max_d=1e99;

	/* initialize pairs of index and distance */
	for (int i = 0; i < knn; i++) {
		nn_d[i] = 1e99-i;
	}
	
	find_dist_kernel(xtr, xquery, TRAINELEMS, prob_dim, knn, computed_distance); 

	for (int i=0; i<TRAINELEMS; i++){
		if( computed_distance[i] < max_d){
			nn_x[max_i] = i;       
			nn_d[max_i] = computed_distance[i];
			max_d = compute_max_pos( nn_d, knn, &max_i);
		}
	}

	float sum = 0.0;
	for (int i = 0; i < knn; i++)
		sum += ytr[nn_x[i]];

	return  sum / knn;
}

/*======================================== Main Execution ========================================*/

int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}
	
	FILE *fpin = NULL;
	char *trainfile = argv[1];
	char *queryfile = argv[2];
	
	// Allocate memory to the CPU 
	// Allocate memory space for the Training Data.
	static float xtr_h[TRAINELEMS*PROBDIM];
	static float ytr[TRAINELEMS];
	float computed_distance[TRAINELEMS];

	// Allocate memory space for the Query Data.
	float yquery[QUERYELEMS];
	float xquery_h[QUERYELEMS*PROBDIM];

	// Allocate memory to the GPU
	float *xquery_d;
	float *xtr_d;

	// Read Training Data to Host.
	fpin = open_traindata(trainfile);
	for (int i=0;i<TRAINELEMS;i++) {
		for (int k = 0; k < PROBDIM; k++)
			xtr_h[i*PROBDIM+k] = read_nextnum(fpin);
			
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
				xquery_h[i*PROBDIM + j] = read_nextnum(fpin);
			}
			yquery[i] = read_nextnum(fpin);
	}
	fclose(fpin);

	xquery_d = (float *)acc_malloc(QUERYELEMS*PROBDIM*sizeof(float));
	xtr_d = (float *)acc_malloc(TRAINELEMS*PROBDIM*sizeof(float));

	// Copy Data to Device
	acc_memcpy_to_device(xquery_d,xquery_h,QUERYELEMS*PROBDIM*sizeof(float));
	acc_memcpy_to_device(xtr_d,xtr_h,TRAINELEMS*PROBDIM*sizeof(float));
	
	float yp, sse = 0.0;
	float err_sum = 0.0;
	double totalTime = 0;
	
	timer runtime;
    runtime.start();
	for (int i = 0; i < QUERYELEMS; i++) {
		yp = find_knn_value( &xquery_d[i*PROBDIM], PROBDIM, NNBS, xtr_d, ytr, computed_distance);
		
		// Calculate Error
		sse += (yquery[i]-yp)*(yquery[i]-yp);
		err_sum +=  100.0*fabs((yp-yquery[i])/ yquery[i]);
	}
	runtime.stop();
	totalTime = runtime.get_timing();

	// Free memory from the GPU
	acc_free(xquery_d);
	acc_free(xtr_d);
	
	double mse = sse/QUERYELEMS;
	double ymean = compute_mean(yquery, QUERYELEMS);
	double var   = compute_var(yquery, QUERYELEMS, ymean);

	totalTime = totalTime * 1000.0;	// convert to ms

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum/QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", 1-(mse/var));
	printf("Total Time = %lf ms\n",totalTime);
	printf("Τime per QElement = %lf ms\n", totalTime/(QUERYELEMS));

	return 0;
}