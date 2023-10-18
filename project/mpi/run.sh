#!/bin/bash

procs=2

./gendata trainingset.txt queryset.txt
mpirun -n $procs ./knn_mpi trainingset.txt queryset.txt
mpirun -n $procs ./knn_mpi_simd trainingset.txt queryset.txt
