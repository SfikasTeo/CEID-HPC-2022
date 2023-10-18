//  Problem Description: Principal component analysis applied to image compression

// Included libraries :
// Zlibc is a read-only compressed file-system emulation
#include <zlib.h>

#include <iostream> // CPP I/O
#include <cstdio>   // C I/O
#include <cstring>  // C strings usage 
#include <cstdlib>  // C memory allocation
#include <cassert>  // C assert
#include <cmath>    // C math library
#include <omp.h>    // Parallelization
#include <cblas.h>  // C Blas interface
#include <lapacke.h>// C Lapack interface
#include <x86intrin.h> // SIMD

extern __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c);
#define SWAP(x, y) do { typeof(x) SWAP = x; x = y; y = SWAP; } while (0)
#define MANUAL_SIMD 1

void update_execution_parameters(int* m, int* n, int* npc, char* features, char** input, int* threads, int argc, char** argv ){
    // parse input parameters
    if ((argc != 1) && (argc != 13)) {
        std::cout << "Usage: " << argv[0] << " -m <rows> -n <cols> -npc <number of principal components> \
        -if <'input filename'> -features <'r' or 'c'> -threads <int>\n";
        exit(1);
    }
    for(int i = 1; i < argc; i++ ) {
        if( strcmp( argv[i], "-m" ) == 0 ) {
            *m = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-n" ) == 0 ) {
            *n = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-npc" ) == 0 ) {
            *npc = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-if" ) == 0 ) {
            *input = argv[i+1];
            i++;
        }
        if( strcmp( argv[i], "-features" ) == 0 ) {
            *features = *argv[i+1];
            i++;
        }
		if( strcmp( argv[i], "-threads" ) == 0 ) {
            *threads = atoi(argv[i+1]);
            i++;
        }
    }
    if (npc > n) npc = n;
}

// Helper function for reading .bin.gz files
double *read_gzfile(char *filename, int frows, int fcols, int rows, int cols)
{
    double *A, *buf;
    gzFile fp;
    int i;

    A = new (std::nothrow) double[rows*cols];
    assert(A != NULL);
    buf = new (std::nothrow) double[fcols];
    assert(buf != NULL);

    fp = gzopen(filename, "rb");
    if (fp == NULL) {
        std::cout << "Input file not available!\n";
        exit(1);
    }

    for (i = 0; i < rows; i++) {
        gzread(fp, buf, fcols*sizeof(double));
        memcpy(&A[i*cols], buf, cols*sizeof(double));
    }
    gzclose(fp);
    delete[] buf;
    return A;
}

// Helper function for writing matrix to file.
void write_ascii(const char* const filename, const double* const data, const int rows, const int cols)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        std::cout << "Failed to create output file\n";
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%.4lf ", data[i*cols+j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

// Helper function.
void print_mtrx(const double* mtrx, int rows, int columns, const char major){
    assert( major == 'r' || major == 'c' );
    bool order = ( major == 'c')? 1:0;
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            printf("%f\t", (order)? mtrx[j*rows+i]:mtrx[i*columns+j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Matrix transpose function
double *dtranspose(const double *__restrict__ input, const int rows, const int cols ){
    double* output = new (std::nothrow) double[cols*rows];
    assert( output!=NULL );
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            output[i*rows + j] = input[j*cols+i];
        }
    }
    return output;
}

void dinplace_transpose(double **__restrict__ input, const int rows, const int cols ){
    double* output = new (std::nothrow) double[cols*rows];
    assert( output!=NULL );
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            output[i*rows + j] = *input[j*cols+i];
        }
    }
	delete[] *input;
    *input = output;
}

void dfeature_analysis(const double *__restrict__ mtrx, int rows, int columns, double *__restrict__ const mean,\
																double *__restrict__ const std, const char major){
    assert( major == 'r' || major == 'c' );
    if ( major == 'c') SWAP(rows, columns);
    int i, j;
    double sum, sum_squared;
	#pragma omp parallel for private(i,j,sum,sum_squared)
    for ( i = 0; i < rows; i++) {   
        sum = 0, sum_squared = 0;
        for( j = 0; j < columns; j++) {
            sum += mtrx[i*columns + j];
            sum_squared += mtrx[i*columns + j]* mtrx[i*columns + j];
        }
        mean[i] = (sum / columns);
        std[i] = sqrt(( sum_squared - columns*( mean[i]* mean[i])) / (columns-1));
        // Estimator for the standard deviation 
    }
}

void dnormalize_mtrx(double *__restrict__ const mtrx, int rows, int columns, const double *__restrict__ const mean,\
															const double *__restrict__ const std, const char major){
    assert( major == 'r' || major == 'c' );
    if ( major == 'c') SWAP(rows, columns);
    int i,j;
	#pragma omp simd collapse(2)
    for ( i = 0; i < rows; i++) {   
        for( j = 0; j < columns; j++) {
            mtrx[i*columns+j] = ( mtrx[i*columns+j] - mean[i] ) / std[i];
        }
    }
}

void dnormalize_mtrx_manualSIMD(double *__restrict__ const mtrx, int rows, int columns, const double *__restrict__ const mean,\
															const double *__restrict__ const std, const char major){
    assert( major == 'r' || major == 'c' );
    if ( major == 'c') SWAP(rows, columns);
	__m256d meanV,stdV,mtrxV,output; 
    int i,j,remaining;
    for ( i = 0; i < rows; i++) {
		meanV = _mm256_set1_pd(mean[i]);
		stdV  = _mm256_set1_pd(std[i]);
        for( j = 0; j+4 < columns; j+=4 ) {
			mtrxV = _mm256_loadu_pd(&mtrx[i*columns+j]);
			output = _mm256_div_pd( _mm256_sub_pd(mtrxV,meanV),stdV);
            _mm256_storeu_pd(&mtrx[i*columns+j],output);
        }
		remaining = columns-(int)columns/4 * 4;
            switch ( remaining ) {
                case 3:
					mtrx[i*(columns)-1] = (mtrx[i*(columns)-1] - mean[i]) / std[i];
					mtrx[i*(columns)-2] = (mtrx[i*(columns)-2] - mean[i]) / std[i];
					mtrx[i*(columns)-3] = (mtrx[i*(columns)-3] - mean[i]) / std[i]; 
				break;
                case 2:
					mtrx[i*(columns)-1] = (mtrx[i*(columns)-1] - mean[i]) / std[i];
					mtrx[i*(columns)-2] = (mtrx[i*(columns)-2] - mean[i]) / std[i];
				break;
                case 1:
					mtrx[i*(columns)-1] = (mtrx[i*(columns)-1] - mean[i]) / std[i];
				break;
            }
    }
}

void ddenormalize_mtrx(double *__restrict__ const mtrx, int rows, int columns, const double *__restrict__ const mean,\
																const double *__restrict__ const std, const char major){
    assert( major == 'r' || major == 'c' );
    if ( major == 'c') SWAP(rows, columns);
    int i,j;
	#pragma omp simd collapse(2)
    for ( i = 0; i < rows; i++) {   
        for( j = 0; j < columns; j++) {
            mtrx[i*columns+j] = ( mtrx[i*columns+j] * std[i] ) + mean[i];
        }
    }
}

void ddenormalize_mtrx_manualSIMD(double *__restrict__ const mtrx, int rows, int columns, const double *__restrict__ const mean,\
																const double *__restrict__ const std, const char major){
    assert( major == 'r' || major == 'c' );
    if ( major == 'c') SWAP(rows, columns);
	__m256d meanV,stdV,mtrxV,output; 
    int i,j,remaining;
    for ( i = 0; i < rows; i++) {
		meanV = _mm256_set1_pd(mean[i]);
		stdV  = _mm256_set1_pd(std[i]);
        for( j = 0; j+4 < columns; j+=4 ) {
			mtrxV = _mm256_loadu_pd(&mtrx[i*columns+j]);
			output = _mm256_fmadd_pd(mtrxV,stdV,meanV);
            _mm256_storeu_pd(&mtrx[i*columns+j],output);
        }
		remaining = columns-(int)columns/4 * 4;
            switch ( remaining ) {
                case 3:
					mtrx[i*(columns)-1] = mtrx[i*(columns)-1]*std[i]+mean[i];
					mtrx[i*(columns)-2] = mtrx[i*(columns)-2]*std[i]+mean[i];
					mtrx[i*(columns)-3] = mtrx[i*(columns)-3]*std[i]+mean[i]; 
					break;
                case 2:
					mtrx[i*(columns)-1] = mtrx[i*(columns)-1]*std[i]+mean[i];
					mtrx[i*(columns)-2] = mtrx[i*(columns)-2]*std[i]+mean[i];
					break;
                case 1:
					mtrx[i*(columns)-1] = mtrx[i*(columns)-1]*std[i]+mean[i];
					break;
            }
    }
}

double *dcovariance_mtrx(const double *__restrict__ mtrx, int rows, int columns, const char major){
    assert( major == 'r' || major == 'c' );
    if ( major == 'r') SWAP(rows, columns);
    int i, j, k; double sum;
    double* Covariance = new (std::nothrow) double[columns*columns]; 
    assert( Covariance!=NULL );
	#pragma omp parallel for
    for ( i = 0; i < columns; i++){
        for ( j = 0; j < columns; j++){
            sum = 0;
            for ( k = 0; k < rows; k++){
                sum += mtrx[i*rows+k] * mtrx[j*rows+k];
            }
            Covariance[i*columns+j] = sum / (double)(rows-1);
        }
    }
    return Covariance;
}

double *dgemm(	const double *A, const char transA, int rows_opA, int columns_opA, \
				const double alpha, const double(*alpha_op)(const double op1, const double op2),\
				const double *B, const char transB, int columns_opB, \
				const char major, const int blockSize){
	if (transA == 't') A = dtranspose(A,columns_opA,rows_opA);
	if (transB == 't') B = dtranspose(B,columns_opB,columns_opA);
	double* C = new (std::nothrow) double[rows_opA*rows_opA];
	assert( C!= NULL );
    
	// In order to optimize this code blocking on the bases
	// of a cache level size should be implemented and 
	// parallelize each block == Each thread is associated 
	// with the same number of rows(A) and cols(B) depending
	// on the cache level sizes and number of procs in use.
	// In order for this to be effective in terms of cache
	// misses transposing the second matrix might actually
	// be advisable, especially for gpus. 

	int i,j,k,ijsum;
	for( i=0; i<rows_opA; i++){
		for( j=0; j<columns_opB; j++){
			ijsum = 0;
			for (k=0;k<columns_opA; k++){
				ijsum += A[i*rows_opA+k] * B[j*columns_opB+k];
			}
			C[i*columns_opA+j] = (*alpha_op)(ijsum,alpha);
		}
	}
	return C;
}

inline double mul(const double operant1, const double operant2){
	return operant1*operant2;
}

inline double div(const double operant1, const double operant2){
	return operant1/operant2;
}

double *dsplit_mtrx_cmajor(const double *__restrict__ mtrx, int rows, int columns, int keep_columns){
    double* submtrx = new (std::nothrow) double[rows*keep_columns]; assert( submtrx!=NULL);
    int i,j,k;
    for ( i = (columns-keep_columns), k = 0; i < columns; i++, k++ ){
        for ( j = 0; j < rows; j++)
            submtrx[k*rows+j]  = mtrx[i*rows+j];
    }
    return submtrx;
}

double *dsplit_mtrx_rmajor(const double *mtrx, int rows, int columns, int keep_columns){
    double* submtrx = new (std::nothrow) double[rows*keep_columns]; assert( submtrx!=NULL);
    int i,j,k;
    for ( i = 0; i < rows; i++){
        for ( j = (columns-keep_columns), k=0; j < columns; j++, k++)
            submtrx[i*keep_columns+k] = mtrx[i*columns+j];
    }
    return submtrx;
}

///////////////////////////////////////////////////////////////////////////////
//
// elvis.bin.gz:   469x700
// cyclone.bin.gz: 4096x4096
// earth.bin.gz:   9500x9500
//
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//
// export OPENBLAS_NUM_THREADS= #Threads
// export GOTO_NUM_THREADS= #Threads
// export OMP_NUM_THREADS= #Threads
//
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    // IF the opeblas library was build with 
    // USE_OPENMP = 1 we nudge the threading 
    // to follow the number of threads set by
    // the openmp fucntion calls
	int threads = 1;		// Default Thread Value
    double t_elapsed;       // Variable for evalutating the elapsed time.

    // input parameters (default)
    char features = 'r', major = 'r';    // The major of the starting matrix
    int m = 469, n = 700;                // image size (rows, columns)
    int npc = 50;                        // number of principal components
    char *input  = (char *)"elvis";      // input filename (compressed binary)
    char *input_bin, *inputFilePath, *output;
    update_execution_parameters(&m,&n,&npc,&features,&input,&threads,argc,argv);
	
	omp_set_num_threads( threads );
    omp_set_dynamic(0);     // Force the number of generated threads.
    
    // Structure the input and output parameters
    assert(asprintf(&input_bin,"%s%s", input, ".bin.gz"));
    assert(asprintf(&inputFilePath,"%s%s", "../pca_data/", input_bin));
    assert(asprintf(&output,"%s%s", input,"_reconstructed.bin"));

    ///////////////////////////////////////////////////////////////////////////
    // Read image data.  The image dimension is m x n.  The returned pointer
    // points to the data in row-major order.  That is, if (i,j) corresponds to
    // to the row and column index, respectively, you access the data with
    // pixel_{i,j} = I[i*n + j], where 0 <= i < m and 0 <= j < n.
    ///////////////////////////////////////////////////////////////////////////

    double* Image_mxn = read_gzfile(inputFilePath, m, n, m, n);

    ///////////////////////////////////////////////////////////////////////////
    // It is important to be decided whether the Rows or the Columns of the
    // matrix that represents the input are considered the features of the image
    // in use. The matrix either way is stored in row major order. Depending on
    // the chosen features, transposing the matrix may be helpfull so that the
    // features are stored sequentially. This means that if the features are the
    // columns the matrix should be transposed so that it is stored in column
    // major order.
    ///////////////////////////////////////////////////////////////////////////

    if ( features == 'c' ){        
		double* temp = dtranspose( Image_mxn, m, n );
        delete[] Image_mxn; Image_mxn = temp; 
        major = 'c';
    }
    
    double start_t = omp_get_wtime();
    ///////////////////////////////////////////////////////////////////////////
    // 1. Compute mean and standard deviation of your image features
    t_elapsed = -omp_get_wtime();

    double* Mean = new (std::nothrow) double[n]; assert(Mean != NULL);
    double* Std  = new (std::nothrow) double[n]; assert(Std  != NULL);
    dfeature_analysis( Image_mxn, m, n, Mean, Std, major);

    t_elapsed += omp_get_wtime();
    std::cout << "MEAN/STD TIME=\t" << t_elapsed << " seconds\n";

    ///////////////////////////////////////////////////////////////////////////
    // 2. Normalize the data
    t_elapsed = -omp_get_wtime();

	#if MANUAL_SIMD
    	dnormalize_mtrx_manualSIMD( Image_mxn, m, n, Mean, Std, major);
	#else
		dnormalize_mtrx( Image_mxn, m, n, Mean, Std, major);
	#endif

    t_elapsed += omp_get_wtime();
    std::cout << "NORMAL. TIME=\t" << t_elapsed << " seconds\n";

    ///////////////////////////////////////////////////////////////////////////
    // 3. Build the covariance matrix
    t_elapsed = -omp_get_wtime();
    
    double* Covariance_nxn = dcovariance_mtrx(Image_mxn, m, n, major);

    t_elapsed += omp_get_wtime();
    std::cout << "C-MATRIX TIME=\t" << t_elapsed << " seconds\n";

    ///////////////////////////////////////////////////////////////////////////
    // 4. Compute the eigenvalues and eigenvectors of the covariance matrix.
    //    Use LAPACK here.
    t_elapsed = -omp_get_wtime();

    double* EigenValues_1xn = new (std::nothrow) double[n]; 
    assert( EigenValues_1xn != NULL);

    LAPACKE_dsyev( (major=='c')?LAPACK_COL_MAJOR:LAPACK_ROW_MAJOR, 'V', 'U', n, Covariance_nxn, n, EigenValues_1xn);
    t_elapsed += omp_get_wtime();
    std::cout << "DSYEV TIME=\t" << t_elapsed << " seconds\n";

    ///////////////////////////////////////////////////////////////////////////
    // 5. Compute the principal components and report the compression ratio
    t_elapsed = -omp_get_wtime();
    
    double* ReducedEigenVectors_nxnpc = (major=='c')? dsplit_mtrx_cmajor( Covariance_nxn, n, n, npc ) \
    : dsplit_mtrx_rmajor(Covariance_nxn, n, n, npc);

    double* PCReduced_mxnpc = new (std::nothrow) double[m*npc];
    assert(PCReduced_mxnpc != NULL);

    cblas_dgemm( (major=='c')?CblasColMajor:CblasRowMajor, CblasNoTrans, CblasNoTrans, m, npc, n, 1, Image_mxn, m, \
    ReducedEigenVectors_nxnpc, n,  0, PCReduced_mxnpc, m );

    t_elapsed += omp_get_wtime();
    std::cout << "PCREDUCED TIME=\t" << t_elapsed << " seconds\n";

    #ifdef _DEBUG_
        char *compressedFile, *principalComponents, *eigenVectors, *rmean, *rstd;
        assert(asprintf(&principalComponents,"%s%s",input,"_principalComponents.bin"));
        assert(asprintf(&eigenVectors,"%s%s",input,"_eigenVectors.bin"));
        assert(asprintf(&rmean,"%s%s",input,"_meanValues.bin"));
        assert(asprintf(&rstd,"%s%s",input,"_stdValues.bin"));
        write_ascii(eigenVectors,ReducedEigenVectors_nxnpc,n,npc);
        write_ascii(principalComponents,PCReduced_mxnpc,m,npc);
        write_ascii(rmean, Mean, n, 1);
        write_ascii(rstd, Std, n, 1);
        asprintf(&compressedFile,"%s %s%s %s %s %s %s %s","tar -czf",input,".tar.gz", \
        principalComponents, eigenVectors, rmean, rstd, "--remove-files");
        system(compressedFile);
        free(compressedFile), free(principalComponents), free(eigenVectors);
        free(rmean), free(rstd);
    #endif

    double end_t = omp_get_wtime();
    std::cout << "OVERALL TIME=\t" << end_t - start_t << " seconds\n";

    ///////////////////////////////////////////////////////////////////////////
    // 6. Reconstruct the image from the compressed data and dump the image in
    //    ascii.

    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasTrans, m, n, npc, 1, PCReduced_mxnpc, m, ReducedEigenVectors_nxnpc, n, 0, Image_mxn, m);
	
	#if MANUAL_SIMD
		ddenormalize_mtrx_manualSIMD( Image_mxn , m, n, Mean, Std, major);
	#else
    	ddenormalize_mtrx( Image_mxn , m, n, Mean, Std, major);
	#endif

    // Write the reconstructed image in ascii format. You can view the image
    // in Matlab with the show_image.m script.
    if ( features == 'c' ){
        double* temp = dtranspose( Image_mxn, n, m );
		#if !MANUAL_SIMD
			// I cannot understand this in any
			// way shape or form. With the manual
			// SIMD functions this leads to either
			// double free or corruption (out),
			// or munmap_chunk(): invalid pointer,
			// Error messages -> I cant find any 
			// correlation between this delete 
			// statement and the manual SIMD codes
			delete[] Image_mxn;
			// This is most propably a memory leak
			// If the omp simd function are userd
			// The code will execute correctly
			// making the error even more confusing.
			// The manual functions do nothing more
			// Than utilize simd instruction sets
		#endif
        Image_mxn = temp; 
	}

    write_ascii(output, Image_mxn, m, n);
    ///////////////////////////////////////////////////////////////////////////

    // cleanup
    free(input_bin), free(inputFilePath), free(output);
    delete[] Image_mxn;
    delete[] ReducedEigenVectors_nxnpc;
    delete[] EigenValues_1xn;
    delete[] PCReduced_mxnpc;
    delete[] Mean;
    delete[] Std;

    return 0;
}
