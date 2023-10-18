#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

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

//======================================== Device Functions ========================================//
	
__global__ void find_dist_kernel(const float * __restrict__ xquery_d, const float * __restrict__ xtr_d, const int prob_dim, float * computed_dist_d, const int qi )
{
	// Each thread represents one training element and computes
	// the distance between the query element and the training element.
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float xquery_shared[PROBDIM];
	
	// Load the query element in shared memory for faster access.
	// Underutilization of Threads in case of prob_dim < threads
	// number of a block. Minimum number of threads per block =
	// Warp size = 32. In case of prob_dim = 16 -> 50% utilization.
	if ( threadIdx.x < prob_dim )
		xquery_shared[threadIdx.x] = xquery_d[qi*prob_dim+threadIdx.x];

	__syncthreads();
	
	// Compute distance
	int i;
	float s = 0.0;
	#pragma unroll
	for (i = 0; i < prob_dim; i++) {
		s += pow( xquery_shared[i]- xtr_d[index*prob_dim+i],2);
	}
	s = sqrt(s);

	computed_dist_d[index] = s;	
}

float find_knn_value( 	const	float * __restrict__ xquery_d, const float * __restrict__ ytr_h, const float * __restrict__ xtr_d, \
						const int prob_dim, const int knn, const int qi, float * __restrict__ computed_dist_h, float *computed_dist_d )
{
	// initialize pairs of index and distance //
	int i, nn_x[MAX_NNB] = {-1}, max_i=0;
	float nn_d[MAX_NNB], max_d=1e99;

	for (int i=0; i < knn; i++) {
		nn_d[i] = 1e99-i;
	}
		
	unsigned int threadsize = 128;
	unsigned int blocksize = TRAINELEMS/threadsize;

	find_dist_kernel<<<blocksize,threadsize>>>(xquery_d, xtr_d, prob_dim, computed_dist_d, qi);
	cudaMemcpy(computed_dist_h,computed_dist_d,TRAINELEMS*sizeof(float), cudaMemcpyDeviceToHost);
	
	for ( i=0; i<TRAINELEMS; i++){
		if( computed_dist_h[i] < max_d){
			nn_x[max_i] = i;       
			nn_d[max_i] = computed_dist_h[i];
			max_d = compute_max_pos( nn_d, knn, &max_i);
		}
	}
	
	// predict value
	float sum = 0.0;
	for ( i = 0; i < knn; i++)
		sum += ytr_h[nn_x[i]];

	return sum/knn;
}

//======================================== Main Execution ========================================//

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
	static float ytr_h[TRAINELEMS];
	
	// Allocate memory space for the Query Data.
	float xquery_h[QUERYELEMS*PROBDIM];
	float yquery_h[QUERYELEMS];

	float *computed_dist_h;
	cudaHostAlloc((void **) &computed_dist_h, TRAINELEMS*sizeof(float),cudaHostAllocDefault);

	// Allocate memory to the GPU
	float *xquery_d;
	float *xtr_d;
	float *computed_dist_d;

	cudaMalloc((void **) &xquery_d, QUERYELEMS*PROBDIM*sizeof(float));
	cudaMalloc((void **) &xtr_d, TRAINELEMS*PROBDIM*sizeof(float));
	cudaMalloc((void **) &computed_dist_d, TRAINELEMS*sizeof(float));

	// Read Training Data to Host.
	fpin = open_traindata(trainfile);
	for (int i=0;i<TRAINELEMS;i++) {
		for (int k = 0; k < PROBDIM; k++)
			xtr_h[i*PROBDIM+k] = read_nextnum(fpin);
			
			#if defined(SURROGATES)
				ytr_h[i] = read_nextnum(fpin);
			#else
				ytr_h[i] = 0;
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
			yquery_h[i] = read_nextnum(fpin);
	}
	fclose(fpin);

	// Copy Data to Device
	cudaMemcpy(xquery_d,xquery_h,QUERYELEMS*PROBDIM*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(xtr_d,xtr_h,TRAINELEMS*PROBDIM*sizeof(float), cudaMemcpyHostToDevice);

	float sse = 0.0, yp;
	float err_sum = 0.0;
	double totalTime = 0;
	
	timer runtime;
    runtime.start();
	for(int i = 0; i < QUERYELEMS; i++)	{
		yp= find_knn_value( xquery_d, ytr_h, xtr_d, PROBDIM, NNBS, i, computed_dist_h, computed_dist_d );

		// Calculate Error
		sse += (yquery_h[i]-yp)*(yquery_h[i]-yp);
		err_sum +=  100.0*fabs((yp-yquery_h[i])/yquery_h[i]);
	}
	runtime.stop();
	totalTime = runtime.get_timing();
	
	// Free memory from the GPU
	cudaFree(computed_dist_d);
	cudaFree(xquery_d);
	cudaFree(xtr_d);

	// Free memory From the CPU
	cudaFreeHost(computed_dist_h);

	float mse = sse/QUERYELEMS;
	float ymean = compute_mean(yquery_h, QUERYELEMS);
	float var   = compute_var(yquery_h, QUERYELEMS, ymean);
	
	totalTime = totalTime * 1000.0;	// convert to ms

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum/QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", 1-(mse/var));
	printf("Total Time = %lf ms\n",totalTime);
	printf("Τime per QElement = %lf ms\n", (totalTime/QUERYELEMS));
	
	return 0;
}
