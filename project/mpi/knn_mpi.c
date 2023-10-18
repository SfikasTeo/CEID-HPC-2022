#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

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

#ifdef DEBUG
	double timer1=0.0,timer2=0.0;
	int computeMaxPositionLoopCounter = 0;
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

	#ifdef DEBUG
		double t1,t2,t3,t4;
	#endif

	// Brute force -> find the knn 
	for (i = 0; i < TRAINELEMS; i++) {
		#ifdef DEBUG
			t1 = gettime();
		#endif
		new_d = compute_euclidean_distance(xquery, &xtr[i*prob_dim], prob_dim);
		#ifdef DEBUG
			t2 = gettime();	timer1+= t2-t1;

			t3 = gettime();
		#endif
		if (new_d < max_d) {
			#ifdef DEBUG
				computeMaxPositionLoopCounter++;
			#endif
			nn_x[max_i] = i;
			nn_d[max_i] = new_d;
			max_d = compute_max_position(nn_d, knn, &max_i);
		}
		#ifdef DEBUG
			t4 = gettime();	timer2+= t4-t3;
		#endif
	}
		
	// predict value
	double sum = 0.0;
	for (int i = 0; i < knn; i++)
		sum += ytr[nn_x[i]];

	return sum/knn;
}


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

	double t_sum = 0.0;
	double sse = 0.0;
	double err_sum = 0.0;

	// Initialize MPI	
	int rank, procs;
	MPI_Init(&argc,&argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

	int step = QUERYELEMS/procs;
	int start = rank * step;
	int end = (rank + 1) * step;
	if(rank == procs -1) end = QUERYELEMS;

	int counter = 0;
	double yp;
	double t_start = gettime();
	for (int i=start; i<end; i++) {
		counter++;
		// Compute the knn and approximate query
		yp = find_knn_value( &xquery[i*PROBDIM], ytr, xtr, PROBDIM, NNBS );
		// Calculate Error
		sse += (yquery[i]-yp)*(yquery[i]-yp);
		err_sum +=  100.0*fabs((yp-yquery[i])/yquery[i]);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double t_end = gettime();

	#ifdef DEBUG
		printf("Proc [%d] Iterations: [%d] Total Time [%lf] s  Timer1 [%lf] s Timer2 [%lf] s\n",rank,counter,t_end-t_start,timer1,timer2);
	#endif

    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &sse, &sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &err_sum, &err_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(rank == 0)
	{
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
	}

	MPI_Finalize();
	return 0;
}