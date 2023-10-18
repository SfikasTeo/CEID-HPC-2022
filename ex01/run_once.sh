#!/bin/bash

# Execution arguments : D L N T dt CheckPoint_ChunkSize Resume_Signal Hybric_Threads
D=1
L=1
T=1000
N=2048
dt=1e-9
Ch=100000         # If Ch > T -> NO checkpoints.
Si=0
Th=3

P=1
#mpirun -n $P ./diffusion2d_mpi_nb $D $L $N $T $dt                > mpi_nb.$P.$T.$N.txt
#mpirun -n $P ./diffusion2d_mpi_nb_2 $D $L $N $T $dt $Ch $Si      > mpi_nb_2.$P.$T.$N.txt
#mpirun -n $P ./diffusion2d_mpi_nb_2_hybrid $D $L $N $T $dt $Ch $Si $Th  > mpi_nb_2_hybrid.$P.$T.$N.txt

P=4
mpirun -n $P ./diffusion2d_mpi_nb $D $L $N $T $dt                > mpi_nb.$P.$T.$N.txt
mpirun -n $P ./diffusion2d_mpi_nb_2 $D $L $N $T $dt $Ch $Si      > mpi_nb_2.$P.$T.$N.txt
mpirun -n $P ./diffusion2d_mpi_nb_2_hybrid $D $L $N $T $dt $Ch $Si $Th  > mpi_nb_hybrid.$P.$T.$N.txt




