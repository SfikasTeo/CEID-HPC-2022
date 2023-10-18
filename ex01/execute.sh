#!/bin/bash

# Execution arguments : D L N T dt CheckPoint_ChunkSize Resume_Signal Hybric_Threads
D=1
L=1
T=10000
N=1024
dt=1e-9         
Ch=100000       # If Ch > T -> NO checkpoints.
Si=0            # Signal For Resuming from Checkpoint    
Th=3            # Threads for Hybrid Parallelization

# Script arguments : end
end=3

Proc=1
Dim=$N
# Execute with 1 process: N = 1024, 2048, 4096
for (( i=0; i<$end; i=$(($i+1)) ))
do  
    mpirun -n $Proc ./diffusion2d_mpi_nb $D $L $Dim $T $dt                    > "diffusion2d_mpi_nb.$Proc.$T.$Dim.txt"
    mpirun -n $Proc ./diffusion2d_mpi_nb_2 $D $L $Dim $T $dt $Ch $Si          > "diffusion2d_mpi_nb_2.$Proc.$T.$Dim.txt"
    mpirun -n $Proc ./diffusion2d_mpi_nb_2_hybrid $D $L $Dim $T $dt $Ch $Si $Th      > "diffusion2d_mpi_nb_2_hybrid.$Proc.$T.$Dim.txt"
    Dim=$(($Dim*2))
done

Proc=4
Dim=$N
# Execute with 4 processes: N = 1024, 2048, 4096
for (( i=0; i<$end; i=$(($i+1)) ))
do  
    mpirun -n $Proc ./diffusion2d_mpi_nb $D $L $Dim $T $dt                    > "diffusion2d_mpi_nb.$Proc.$T.$Dim.txt"
    mpirun -n $Proc ./diffusion2d_mpi_nb_2 $D $L $Dim $T $dt $Ch $Si          > "diffusion2d_mpi_nb_2.$Proc.$T.$Dim.txt"
    mpirun -n $Proc ./diffusion2d_mpi_nb_2_hybrid $D $L $Dim $T $dt $Ch $Si $Th      > "diffusion2d_mpi_nb_2_hybrid.$Proc.$T.$Dim.txt"
    Dim=$(($Dim*2))
done
