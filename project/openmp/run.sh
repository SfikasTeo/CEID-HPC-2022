#!/bin/bash

# Third argument is the threads in use.
threads=2

./gendata trainingset.txt queryset.txt
./knn_omp trainingset.txt queryset.txt		$threads
./knn_omp_simd trainingset.txt queryset.txt	$threads