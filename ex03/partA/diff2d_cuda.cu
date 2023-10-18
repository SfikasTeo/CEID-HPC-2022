#include <iostream>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <cassert>
#include <cmath>

// Cuda runtime library may be
// needed if compiling with gcc
#include <cuda_runtime.h>

#include "timer.hpp"
typedef std::size_t size_type;

// Block Size
#define BSZ 16

__global__ void diffusion_kernel(float *rho_tmp, const float * __restrict__ rho, float fac, int N)
{
	// Use shared memory for gpu caching
    __shared__ float rho_shared[BSZ+2][BSZ+2];

    const int i = threadIdx.x;
	const int j = threadIdx.y;

	// Load the First Data
	int flattened_index = j*BSZ + i;
	int I = flattened_index % (BSZ+2); // including padded columns
	int J = flattened_index / (BSZ+2); // including padded rows
    
	// accross all blocks for padded block
	int global_index = blockIdx.y*(BSZ)*N + blockIdx.x*(BSZ) + J*N + I;
	rho_shared[I][J] = rho[global_index];

	// Load the Remaining Data	
	flattened_index = BSZ*BSZ + j*BSZ +i;
	I = flattened_index % (BSZ+2);
	J = flattened_index / (BSZ+2);
	
	global_index = blockIdx.y*(BSZ)*N + blockIdx.x*(BSZ) + J*N + I;
	if( global_index>= N*N ){return ;}

	if( (I <(BSZ+2)) && (J<(BSZ+2)) && (flattened_index < N*N))
		rho_shared[I][J] = rho[global_index];

	global_index = blockIdx.y*(BSZ)*N + blockIdx.x*(BSZ) + (j+1)*N + i+1;
	if( global_index>= N*N ){return ;}
	
	__syncthreads();
	rho_tmp[global_index]  =
                rho_shared[i+1][j+1]
                + fac * (
                rho_shared[i+2][j+1] +
                rho_shared[i][j+1] +
                rho_shared[i+1][j+2] +
                rho_shared[i+1][j] -
                4*rho_shared[i+1][j+1]
                );
}

class Diffusion2D
{

public:
	// Class constructor
    Diffusion2D(
                const float D,
                const float rmax,
                const float rmin,
                const size_type N
                )
    : D_(D)
    , rmax_(rmax)
    , rmin_(rmin)
    , N_(N)
    , N_tot(N*N)
    , d_rho_(0)
    , d_rho_tmp_(0)
    , rho_(0)
    {
        /// real space grid spacing
        dr_ = (rmax_ - rmin_) / (N_ - 1);

        /// dt < dx*dx / (4*D) for stability
        dt_ = dr_ * dr_ / (6 * D_);

        /// stencil factor
        fac_ = dt_ * D_ / (dr_ * dr_);    

		// Allocate memory on CPU
        rho_ = (float*)malloc(N_tot*sizeof(float));

		// Allocate d_rho_ and d_rho_tmp_ on the GPU and set them to zero
        cudaMalloc((void **)&d_rho_, N_tot*sizeof(float));
        cudaMalloc((void **)&d_rho_tmp_, N_tot*sizeof(float));
		
		// First initialization may not be neccesary
        cudaMemset(d_rho_,0,N_tot*sizeof(float));
        cudaMemset(d_rho_tmp_,0,N_tot*sizeof(float));

        InitializeSystem();
    }

    ~Diffusion2D()
    {
		free(rho_);
        cudaFree(d_rho_);
        cudaFree(d_rho_tmp_);
    }
    
    void PropagateDensity(int steps);
    float GetMoment() {
        cudaMemcpy(rho_,d_rho_,N_tot*sizeof(float),cudaMemcpyDeviceToHost);

        float sum = 0;
        for(size_type i = 0; i < N_; ++i)
            for(size_type j = 0; j < N_; ++j) {
                float x = j*dr_ + rmin_;
                float y = i*dr_ + rmin_;
                sum += rho_[i*N_ + j] * (x*x + y*y);
            }

        return dr_*dr_*sum;
    }
    float GetTime() const {return time_;}
    void WriteDensity(const std::string file_name) const;

private:
	// private member functions
    void InitializeSystem();

	// private member variables
    const float D_, rmax_, rmin_;
    const size_type N_;
    size_type N_tot;

    float dr_, dt_, fac_;

    float time_;
    float *d_rho_, *d_rho_tmp_;
    float *rho_;
};

void Diffusion2D::WriteDensity(const std::string file_name) const
{
    cudaMemcpy(rho_,d_rho_,N_tot*sizeof(float),cudaMemcpyDeviceToHost);
    
    std::ofstream out_file;
    out_file.open(file_name.c_str(), std::ios::out);
    if(out_file.good()){
        for(size_type i = 0; i < N_; ++i){     
            for(size_type j = 0; j < N_; ++j)
                out_file << (i*dr_+rmin_) << '\t' << (j*dr_+rmin_) << '\t' << rho_[i*N_ + j] << "\n";

            out_file << "\n";
        }
    }
    out_file.close();
}

void Diffusion2D::PropagateDensity(int steps)
{
    using std::swap;
    // Dirichlet boundaries; central differences in space, forward Euler in time

    int block_dim_x = BSZ; // block dimension x
    int block_dim_y = BSZ; // block dimension y
    int grid_dim_x = (N_ + block_dim_x -1) / block_dim_x;
    int grid_dim_y = (N_ + block_dim_y -1) / block_dim_y;
    const dim3 blockSize(block_dim_x,block_dim_y);
    const dim3 gridSize(grid_dim_x,grid_dim_y);
        
    for(int s = 0; s < steps; ++s)
    {
	    diffusion_kernel<<< gridSize , blockSize >>>(d_rho_tmp_, d_rho_, fac_, N_);
		swap(d_rho_, d_rho_tmp_);
        time_ += dt_;
    }
}

void Diffusion2D::InitializeSystem()
{
    time_ = 0;
    /// initialize rho(x,y,t=0)
    float bound = 1./2;
    for(size_type i = 0; i < N_; ++i){
        for(size_type j = 0; j < N_; ++j){
            if(std::fabs(i*dr_+rmin_) < bound && std::fabs(j*dr_+rmin_) < bound){
                rho_[i*N_ + j] = 1;
            }
            else{
                rho_[i*N_ + j] = 0;
            }

        }
    }
    cudaMemcpy(d_rho_, &rho_[0], N_tot* sizeof(float), cudaMemcpyHostToDevice);
}

int main(int argc, char* argv[])
{

    if(argc != 2)
    {   // matrix Size N
        std::cerr << "usage: " << argv[0] << " <log2(size)>" << std::endl;
        return 1;
    }

    const float D = 1;
    const float tmax = 0.01;
    const float rmax = 1;
    const float rmin = -1;

    const size_type N_ = 1 << std::atoi(argv[1]);
    const int steps_between_measurements = 100;

    Diffusion2D System(D, rmax, rmin, N_);

    float time = 0;
    timer runtime;
    runtime.start();
    while(time < tmax){
        System.PropagateDensity(steps_between_measurements);
        time = System.GetTime();
		#ifdef DEBUG
        	float moment = System.GetMoment();
        	std::cout << time << '\t' << moment << '\n';
		#endif
    }
    runtime.stop();
    double elapsed = runtime.get_timing();

    std::cerr << argv[0] << "\t N=" << N_ << "\t time=" << elapsed << "s" << std::endl;

    std::string density_file = "Density.dat";
    System.WriteDensity(density_file);

    return 0;
}
