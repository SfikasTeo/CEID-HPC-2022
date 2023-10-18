#include <stdio.h>      // I/O
#include <math.h>       // sqrt
#include <stdlib.h>     // posix_malloc , calloc etc
#include <string.h>     // sprintf
#include <mpi.h>        // MPI
#include <omp.h>        // OMP
#include <immintrin.h>  // AVX, AVX2, FMA

#define _PERF_ 1
#define _MANUAL_SIMD_ 0

// The mean time of central submatrix update time.
double meanTime_CMU = 0;
// The mean time for waiting the transfer of Ghost C.
double meanTime_WGC = 0;

typedef struct Diagnostics_s
{
	double time;        // Timestamp
	double heat;        // General Heat value of the system
} Diagnostics;

typedef struct Diffusion2D_s
{
	double D_, L_, T_;     // Problem arguments, T = advance function iterations.
	int N_;                // N = dimensions of starting matrix R(n*n)
	int Ntot_;             // Ntot = number of cells per process = LocalN*LocalN
	int local_N_;          // Local_N_ = dimensions of submatrices
	int real_N_;           // The Real Dimensions of the matrix + padded "ghost cells"
	double dr_, dt_, fac_; // Problem arguments
	int rank_, procs_;     // Process identifications
	int divisions;         // Number of Divisions of the generalized matrix.
	double *rho_, *rho_tmp_; // Heat value of each cell
	Diagnostics *diag_;
} Diffusion2D;


void write_density_values(double *tmp_rho, char * filename, int procs, int rank_, int Ntot_)
{
	// write your rho_ to file with exscan function for pointers.
	// asychronous write
	MPI_File file;
	if( MPI_File_open( MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file) != MPI_SUCCESS )
	{
		printf("[MPI Process %d] Failed at opening the file.\n", rank_); 
		MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
	}
	
	//prealocate file size
	MPI_File_preallocate( file, Ntot_*procs );

	//get the starting position of the file pointer.
	MPI_Offset base;
	MPI_File_get_position( file,&base );

	//calculate the offset of each individual process.
	MPI_Offset offset = rank_ * (Ntot_ * sizeof(double));
	MPI_Status status;

	MPI_File_write_at( file, base+offset, tmp_rho, Ntot_, MPI_DOUBLE, &status);
	MPI_File_close(&file);
}

void read_density_values(Diffusion2D *D2D,const char * filename)
{
	// initilaze_density like function that reads from a binary file.
	MPI_File file;
	if( MPI_File_open( MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS )
	{
		printf("[MPI Process %d] Failed at opening the file.\n",D2D->rank_); 
		MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
	}
	
	//get the starting position of the file pointer.
	MPI_Offset base;
	MPI_File_get_position( file, &base );

	//calculate the offset of each individual process.
	MPI_Offset offset = D2D->rank_ * (D2D->Ntot_ * sizeof(double));
	MPI_Status status;

	MPI_File_read_at_all( file, base+offset, D2D->rho_, D2D->Ntot_, MPI_DOUBLE, &status);
	MPI_File_close(&file);

}

void initialize_density(Diffusion2D *D2D)
{
	int real_N_ = D2D->real_N_;
	int local_N_ = D2D->local_N_;
	double *rho_ = D2D->rho_;
	double dr_ = D2D->dr_;
	double L_ = D2D->L_;
	int rank_ = D2D->rank_;
	int divisions = D2D->divisions;
	int gi;
	int gj;

	/// Initialize rho(x, y, t=0).
	double bound = 0.25 * L_;
	int x = rank_ % divisions;
	int y = rank_ / divisions;

	for (int i = 1; i <= local_N_; ++i)         // From Row -> 1 : row[ (local_N_ + 1) = ( real_N_ -1 )] -> initialize the valueable information.
	{
		gi = y * local_N_ + i;                  // convert local index to global index ( +1 (+i) in the first iteration balances the ghost cells )
		for (int j = 1; j <= local_N_; ++j)     // From column -> 1 : row[ (local_N_ + 1) = ( real_N_ -1 )] -> initialize the valueable information.
		{
			gj = x * local_N_ + j;
			if (fabs((gi - 1) * dr_ - 0.5 * L_) < bound && fabs((gj - 1) * dr_ - 0.5 * L_) < bound)
			{
				rho_[i * real_N_ + j] = 1;
			}
		}
	} 
}

void init(Diffusion2D *D2D,
		  const double D,
		  const double L,
		  const int N,
		  const int T,
		  const double dt,
		  const int rank,
		  const int procs,
		  const int signal,
		  const char* fn,
          const char* executable)
{
	D2D->D_ = D;
	D2D->L_ = L;
	D2D->N_ = N;
	D2D->T_ = T;
	D2D->dt_ = dt;
	D2D->rank_ = rank;
	D2D->procs_ = procs;
	D2D->divisions = (int)sqrt(procs);

	// Real space grid spacing.
	D2D->dr_ = D2D->L_ / (D2D->N_ - 1);

	// Stencil factor.
	D2D->fac_ = D2D->dt_ * D2D->D_ / (D2D->dr_ * D2D->dr_);

	// Number of rows per process.
	D2D->local_N_ = D2D->N_ / D2D->divisions;

	// Actual dimension of a row (+2 for the ghost cells).
	D2D->real_N_ = D2D->local_N_ + 2;

	// Total number of cells.
	D2D->Ntot_ = (D2D->real_N_) * (D2D->real_N_);

	D2D->rho_       = NULL;
	D2D->rho_tmp_   = NULL;
	posix_memalign((void *)&D2D->rho_tmp_,  64, D2D->Ntot_*sizeof(double));
	posix_memalign((void *)&D2D->rho_,      64, D2D->Ntot_*sizeof(double));

	D2D->diag_ = (Diagnostics *)calloc(D2D->T_, sizeof(Diagnostics));

	// Check that the timestep satisfies the restriction for stability.
	if (D2D->rank_ == 0)
		printf(" [ %s ] Timestep from stability condition is %lf\n", executable, D2D->dr_ * D2D->dr_ / (4.0 * D2D->D_));

	if ( !signal ){
		initialize_density(D2D);
	}
	else{
		read_density_values(D2D,fn);
	}
}

void advance(Diffusion2D *D2D, int step , int checkpoint)
{
	int i; // counter 1
	int j; // counter 2
	int real_N_ = D2D->real_N_;
	int local_N_ = D2D->local_N_;
	double *rho_ = D2D->rho_;
	double *rho_tmp_ = D2D->rho_tmp_;
	double fac_ = D2D->fac_;
	int rank_ = D2D->rank_;
	int procs_ = D2D->procs_;
	int divisions = D2D->divisions;

	// Non-blocking MPI
	MPI_Request req[8];
	MPI_Status status[8];
	
	int left_rank = rank_ - 1;
	int right_rank = rank_ + 1;
	int up_rank = rank_ - divisions;
	int down_rank = rank_ + divisions;

	double *right_send_buffer;
	double *right_store_buffer;
	double *left_send_buffer;
	double *left_store_buffer;

	left_send_buffer = calloc(local_N_, sizeof(double));
	left_store_buffer = calloc(local_N_, sizeof(double));

	right_send_buffer = calloc(local_N_, sizeof(double));
	right_store_buffer = calloc(local_N_, sizeof(double));

	// Exchange ALL necessary ghost cells with neighboring ranks.
	// if the process has a block 12 o'clock. - up
	if (up_rank >= 0)
	{
		// only send the local_N_ cells with valueable information
		MPI_Irecv(&rho_[0 * real_N_ + 1], local_N_, MPI_DOUBLE, up_rank, 100, MPI_COMM_WORLD, &req[0]);
		MPI_Isend(&rho_[1 * real_N_ + 1], local_N_, MPI_DOUBLE, up_rank, 100, MPI_COMM_WORLD, &req[1]);
	}
	else
	{
		req[0] = MPI_REQUEST_NULL;
		req[1] = MPI_REQUEST_NULL;
	}

	// if the process has a block 9 o'clock. - left
	if ((rank_ % divisions) > 0)
	{
		for (i = 1; i <= local_N_; i++)
			left_send_buffer[i - 1] = rho_[i * real_N_ + 1];

		MPI_Irecv(left_store_buffer, local_N_, MPI_DOUBLE, left_rank, 100, MPI_COMM_WORLD, &req[2]);
		MPI_Isend(left_send_buffer, local_N_, MPI_DOUBLE, left_rank, 100, MPI_COMM_WORLD, &req[3]);
	}
	else
	{
		req[2] = MPI_REQUEST_NULL;
		req[3] = MPI_REQUEST_NULL;
	}

	// if the process has a block 3 o'clock. - right
	if ((rank_ % divisions) < divisions - 1)
	{
		for (i = 2; i <= local_N_ + 1; i++)
			right_send_buffer[i - 2] = rho_[i * real_N_ - 2];

		MPI_Irecv(right_store_buffer, local_N_, MPI_DOUBLE, right_rank, 100, MPI_COMM_WORLD, &req[4]);
		MPI_Isend(right_send_buffer, local_N_, MPI_DOUBLE, right_rank, 100, MPI_COMM_WORLD, &req[5]);
	}
	else
	{
		req[4] = MPI_REQUEST_NULL;
		req[5] = MPI_REQUEST_NULL;
	}

	// if the process has a block 6 o'clock. - down
	if (down_rank < procs_)
	{
		MPI_Irecv(&rho_[(real_N_ - 1) * real_N_ + 1], local_N_, MPI_DOUBLE, down_rank, 100, MPI_COMM_WORLD, &req[6]);
		MPI_Isend(&rho_[(real_N_ - 2) * real_N_ + 1], local_N_, MPI_DOUBLE, down_rank, 100, MPI_COMM_WORLD, &req[7]);
	}
	else
	{
		req[6] = MPI_REQUEST_NULL;
		req[7] = MPI_REQUEST_NULL;
	}
   
    #if ! _PERF_
        	double t0 = MPI_Wtime();
    #endif

	// Vectorization done by hand using AVX-2 intrinsics.
   	#if _MANUAL_SIMD_
	// Have not parallelized the manual SIMD instructions yet due to the aftermentioned errors.
	
	// Async update of the Central Submatrix.  
	__m256d rho_vec, rho_vec_right, rho_vec_left, rho_vec_down, rho_vec_up;
	__m256d aux_vec_1, aux_vec_2, aux_vec_3, aux_vec_4, aux_vec_5;
	__m256d aux_vec_n4  = _mm256_set1_pd( -4.0 );
	__m256d aux_vec_fac = _mm256_set1_pd( fac_ );
		
	// In order for the data to be loaded correctly memory alignement is necessary.
	// According to the info of /proc/cpuinfo the cache alignement is 64 bits.
	// The use of posix_memalign have been used in the init function to align
	// he memory of rho_ and rho_tmp_.
	// IMPORTANT: We are traversing the array by doubles ( 8 bytes = 64 bits).
	// IF the start of the array is correctly aligned every other index inside of it
	// shall also be correctly aligned.
	for (i = 2; i < local_N_; i++){			// Rows     -> Central Block, we skip the first valueable row.
		for (j = 2; j < local_N_; j+=4){	// Columns  -> Central Block, we skip the first valueable col.
			// Load From memory to AVX registers

            // IMPORTANT !! !! IMPORTANT !! !! IMPORTANT !! -> why the segmentation faults ?.
            // How can rho_vec_right load correctly but not rho_vec_left ? How does this make any sense ??
			rho_vec         = _mm256_load_pd( &rho_[i * real_N_ + j] );
			rho_vec_right   = _mm256_load_pd( &rho_[i * real_N_ + (j + 1)] );
			rho_vec_left    = _mm256_loadu_pd( &rho_[i * real_N_ + (j - 1)] );	// if we change to _mm256_load_pd -> Segmentation Error
			rho_vec_down    = _mm256_load_pd( &rho_[(i + 1) * real_N_ + j] );
			rho_vec_up      = _mm256_loadu_pd( &rho_[(i - 1) * real_N_ + j] );  // if we change to _mm256_load_pd -> Segmentation Error
			
			// Start the Computation using auxilliary/intermediate vectors
			aux_vec_1   = _mm256_add_pd( rho_vec_right, rho_vec_left);
			aux_vec_2   = _mm256_add_pd( rho_vec_down, rho_vec_up);
			aux_vec_3   = _mm256_add_pd( aux_vec_1, aux_vec_2);

			// Use auxilliary vectors with FMA operations ?
			// Although a register that supports SIMD/intrinsics can be reused in an operation.
			// According to the below answer, we are creating a longer critical path and we
			// encounter a bottlenech on latency not throughput. ( reciprocal latency )
			// In truth, I do not think that i totally grasp the notion of FP latency 
			//  -> More reading must take place.
			// But using multiple FP vector accumulators seems like the better practice.
			// https://stackoverflow.com/questions/66260651/mm256-fmadd-ps-is-slower-than-mm256-mul-ps-mm256-add-ps
            // and https://stackoverflow.com/questions/65818232/improving-performance-of-floating-point-dot-product
            // -of-an-array-with-simd/65827668#65827668
			// This i think is in accordance with the utilization of ILP ( instruction level parallelism )
			// While i think i understand ILP in theory, in practice it gets a bit complicated.
			
			// WARNING: For AMD CPUs the ZEN microarchitecture removed support for the FMA4
			// instruction-set, while retaining FMA3 instuctions. ( FOR FMA operations the
			// number does not signify the verion of FMA but the Number of Registers in USE).
			// According to The article of techpowerup the processor will complete the operation.
			// https://www.techpowerup.com/248560/amd-zen-does-support-fma4-just-not-exposed
			// In truth, I do not exactly know wether or not FMA4 is actually used or not.
			// ALTERATION that is supported: _mm256_add_pd(_mm256_mul_pd(aXX, bYY), cZZ)

			aux_vec_4   = _mm256_fmadd_pd( aux_vec_n4, rho_vec, aux_vec_3);
			aux_vec_5   = _mm256_fmadd_pd( aux_vec_4, aux_vec_fac, rho_vec);

			// Store the end result ( aux_vec_2 ) to memory ( rho_tmp_ )
			_mm256_storeu_pd( &rho_tmp_[i * real_N_ + j] , aux_vec_5 );
		}
	}

	// Update the leftover cells of each row.
	// Starting from index 2 and traversing the array be 4 doubles at once due to
	// vectorization should leave some cells at the end of each theoretical "row"
	// unchanged. Due to the fized sizes of N_ for 1024 , 4096 etc there should be no
	// need for checking The number of leftovers in each "row" with a switch statement.

	int left1 = local_N_ -1; // Update the last element of the central submatrix
	int left2 = local_N_ -2; // that was leftover by the SIMD operations in each "row"

	for (i = 2; i < local_N_; i++){

		// Update the last leftover row
		rho_tmp_[i * real_N_ + left1] = rho_[i * real_N_ + left1] +
										fac_ *
											(+rho_[i * real_N_ + (left1 + 1)]
											+ rho_[i * real_N_ + (left1 - 1)]
											+ rho_[(i + 1) * real_N_ + left1]
											+ rho_[(i - 1) * real_N_ + left1]
											- 4. * rho_[i * real_N_ + left1]);
		// Update the previous leftover row
		rho_tmp_[i * real_N_ + left2] = rho_[i * real_N_ + left2] +
										fac_ *
											(+rho_[i * real_N_ + (left2 + 1)]
											+ rho_[i * real_N_ + (left2 - 1)]
											+ rho_[(i + 1) * real_N_ + left2]
											+ rho_[(i - 1) * real_N_ + left2]
											- 4. * rho_[i * real_N_ + left2]);
	}
		

	// Vectorization Done by openmp automatically
	#else
		 
	// Async update of the Central Submatrix.  
	// Important: Collapse(2) as a clause is mandatory in order for the vectorization to take effect. Otherwise,
	// #pragma omp simd clause must be used in the innermost for loop. That would  call the directive more times
	// and possibly increment the overhead of the clause statement.
	#pragma omp simd collapse(2) 
	for (int i = 2; i < local_N_; ++i){		// Rows     -> Central Block, we skip the first valueable row.
		for (int j = 2; j < local_N_; ++j){ // Columns  -> Central Block, we skip the first valueable col.
			rho_tmp_[i * real_N_ + j] = rho_[i * real_N_ + j] +
										fac_ *
											(+rho_[i * real_N_ + (j + 1)]
											+ rho_[i * real_N_ + (j - 1)]
											+ rho_[(i + 1) * real_N_ + j]
											+ rho_[(i - 1) * real_N_ + j]
											- 4. * rho_[i * real_N_ + j]);
		}
	}
	
	// End of central matrix update computation
	#endif

    #if ! _PERF_
       	double t1 = MPI_Wtime();
        meanTime_CMU += t1 - t0;
    #endif

    #if ! _PERF_
       	t0 = MPI_Wtime();
    #endif

	// ensure boundaries have arrived
	MPI_Waitall(8, req, status);

    #if ! _PERF_
       	t1 = MPI_Wtime();
        meanTime_WGC += t1 - t0;
    #endif

	// Update the matrix's left and right ghost cells with data from the auxiliary arrays.
	// Could be parallelized in a hybrid architecture model. ( if Local_N is big enough )
	for (i = 2; i <= local_N_ + 1; i++)
		rho_[i * real_N_ - 1] = right_store_buffer[i - 2];

	for (j = 1; j <= local_N_; j++)
		rho_[j * real_N_] = left_store_buffer[j - 1];

	// free the now unneeded pointers
	free(right_store_buffer);
	free(right_send_buffer);
	free(left_store_buffer);
	free(left_send_buffer);

	// Update the borders of the Local_N * local_N valueable matrix,
	// With the use of the transferred ghost cell information.

	// Update the top and bottom border rows
	for (i = 1; i <= real_N_; i += (local_N_ - 1))
	{ // row
		for (j = 1; j <= local_N_; j++)
		{ // col
			rho_tmp_[i * real_N_ + j] = rho_[i * real_N_ + j] + fac_ *
																	(+rho_[i * real_N_ + (j + 1)] 
																	+ rho_[i * real_N_ + (j - 1)] 
																	+ rho_[(i + 1) * real_N_ + j] 
																	+ rho_[(i - 1) * real_N_ + j] 
																	- 4. * rho_[i * real_N_ + j]);
		}
	}

	// Update the first and last border column -corners
	for (i = 2; i < local_N_; i++)
	{
		for (j = 1; j <= real_N_; j += (local_N_ - 1))
		{
			rho_tmp_[i * real_N_ + j] = rho_[i * real_N_ + j] + fac_ *
																	(+rho_[i * real_N_ + (j + 1)] 
																	+ rho_[i * real_N_ + (j - 1)] 
																	+ rho_[(i + 1) * real_N_ + j] 
																	+ rho_[(i - 1) * real_N_ + j] 
																	- 4. * rho_[i * real_N_ + j]);
		}
	}

	// Swap rho_ with rho_tmp_. This is much more efficient,
	// because it does not copy element by element, just replaces storage
	// pointers.
	double *tmp_ = D2D->rho_tmp_;
	D2D->rho_tmp_ = D2D->rho_;
	D2D->rho_ = tmp_;
}

void compute_diagnostics(Diffusion2D *D2D, const int step, const double t)
{
	int local_N_ = D2D->local_N_;
	double *rho_ = D2D->rho_;
	double dr_ = D2D->dr_;
	int rank_ = D2D->rank_;
	int real_N_ = D2D->real_N_;

	double heat = 0.0;
	for (int i = 1; i <= local_N_; ++i)
		for (int j = 1; j <= local_N_; ++j)
		{
			heat += rho_[i * real_N_ + j] * dr_ * dr_;
		}
	MPI_Reduce(rank_ == 0 ? MPI_IN_PLACE : &heat, &heat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank_ == 0)
	{
		D2D->diag_[step].time = t;
		D2D->diag_[step].heat = heat;
	}
}

void write_diagnostics(Diffusion2D *D2D, const char *filename)
{
	FILE *out_file = fopen(filename, "w");
	for (int i = 0; i < D2D->T_; i++)
		fprintf(out_file, "%f\t%f\n", D2D->diag_[i].time, D2D->diag_[i].heat);
	fclose(out_file);
}

int main(int argc, char *argv[])
{
	if (argc < 8)
	{
		printf("Usage: %s Heat_Domain_Values L N_Matrix_dimensions Timesteps \
		dt CheckPoint_ChunkSize Resume_Signal\n", argv[0]);
		
		MPI_Abort(MPI_COMM_WORLD,1);
	}

	int rank, procs;
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

	const double D 	= atof(argv[1]);
	const double L 	= atoi(argv[2]);
	const int N 	= atoi(argv[3]);
	const int T 	= atoi(argv[4]);
	const double dt = atof(argv[5]);
	const int checkpoint = atoi(argv[6]);
	const int signal  = atoi(argv[7]);
	
	Diffusion2D system;
	char * file_checkpoint = "diffusion2d_mpi_nb_2_checkpoint.bin";

	init(&system, D, L, N, T, dt, rank, procs, signal, file_checkpoint, argv[0]);
	double t0 = MPI_Wtime();
	for (int step = 0; step < T; ++step)
	{
		advance(&system,step,checkpoint);
		if( (step+1) % checkpoint == 0)
		{
			// If the matrix size is much larger. The problem is actually beeing executed for
			// big data, writing to file should be asychronous with the use of tasks.
			// Important: the system.rho_ should be copied first with the use of memcpy as ->
			// memcpy( tmp_system_rho_, system.rho_, system.Ntot_ *sizeof(double));
			// For our data dimensions, writing to file is about 100 times faster that an 
			// advance, IF our testing timers worked correctly. For now the use of tasks 
			// offers no significant advantage. The directive however shall be as follows ->
			// #pragma omp task firstprivate( tmp_system_rho_ )
			write_density_values( system.rho_, file_checkpoint, procs, rank, system.Ntot_ );
		}
		#if ! _PERF_
				compute_diagnostics(&system, step, dt * step);
		#endif
	}
	double t1 = MPI_Wtime();
	if (rank == 0)
		printf("%lf", t1 - t0 );

    #if ! _PERF_
	    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &meanTime_CMU, &meanTime_CMU, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &meanTime_WGC, &meanTime_WGC, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (rank == 0)
		{
            printf("\n[ %s ] The mean time of central submatrix computation = %lf\n", argv[0], meanTime_CMU/T );
            printf("[ %s ] The mean time of waiting the Ghost Cells tranfer = %lf\n", argv[0], meanTime_WGC/T );
		    char diagnostics_filename[256];
		    sprintf(diagnostics_filename, "diagnostics_mpi_nb_2_%d.dat", procs);
		    write_diagnostics(&system, diagnostics_filename);
	    }
    #endif    

	MPI_Finalize();
	return 0;
}
